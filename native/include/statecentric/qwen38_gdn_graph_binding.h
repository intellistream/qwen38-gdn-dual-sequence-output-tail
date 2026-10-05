#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include "statecentric/qwen38_provider_api_v1.h"

namespace statecentric {

// DIRECT captures embed the exact ordered state addresses. COPY captures embed
// private packed buffers and can be replayed against any slots of the same width.
template <typename CapturedSlot, typename CurrentSlot>
bool Qwen38GdnGraphSlotsMatch(std::span<const CapturedSlot> captured,
                            std::span<const CurrentSlot> current,
                            bool direct) noexcept {
  return captured.size() == current.size() &&
         (!direct || std::equal(captured.begin(), captured.end(), current.begin()));
}

// A current DIRECT layout is a zero-copy opportunity for a NEW capture, not an
// instruction to rebind an existing COPY graph. Both layout modes expose row
// addresses. Keep the capture's packed buffers and gather/scatter those rows.
// copy(destination, bytes, source) is supplied by the caller (ACL in production).
template <typename Copy>
void CopyQwen38GdnGraphRows(
    const statecentric_qwen38_gdn_layout_workspace_v1& layout,
    std::size_t batch_size, std::uint64_t conv_bytes,
    std::uint64_t recurrent_bytes, std::uintptr_t packed_conv,
    std::uintptr_t packed_recurrent, bool gather, Copy&& copy) {
  const auto fits = [](std::uintptr_t base, std::uint64_t capacity,
                       std::uintptr_t address, std::uint64_t bytes) {
    return base != 0 && address >= base && bytes > 0 && bytes <= capacity &&
           capacity <= std::numeric_limits<std::uintptr_t>::max() - base &&
           address - base <= capacity - bytes;
  };
  if (layout.size < sizeof(layout) ||
      layout.abi_version != STATECENTRIC_QWEN38_PROVIDER_ABI_V1 ||
      batch_size == 0 || batch_size > 4 || layout.batch_size != batch_size ||
      (layout.binding_mode != STATECENTRIC_QWEN38_GDN_BINDING_COPY_V1 &&
       layout.binding_mode != STATECENTRIC_QWEN38_GDN_BINDING_DIRECT_V1) ||
      layout.required_arena_bytes != layout.arena_bytes ||
      layout.conv_bytes_per_row != conv_bytes ||
      layout.recurrent_bytes_per_row != recurrent_bytes ||
      conv_bytes > std::numeric_limits<std::uintptr_t>::max() / batch_size ||
      recurrent_bytes > std::numeric_limits<std::uintptr_t>::max() / batch_size ||
      !fits(packed_conv, batch_size * conv_bytes, packed_conv, conv_bytes) ||
      !fits(packed_recurrent, batch_size * recurrent_bytes,
            packed_recurrent, recurrent_bytes)) {
    throw std::runtime_error("Qwen3.8 graph GDN copy binding is invalid");
  }
  // Validate every row before enqueueing any transfer, including on failure.
  for (std::size_t row = 0; row < batch_size; ++row) {
    if (!fits(layout.arena, layout.arena_bytes,
              layout.conv_row_addresses[row], conv_bytes) ||
        !fits(layout.arena, layout.arena_bytes,
              layout.recurrent_row_addresses[row], recurrent_bytes)) {
      throw std::runtime_error("Qwen3.8 graph GDN copy row is out of bounds");
    }
  }
  for (std::size_t row = 0; row < batch_size; ++row) {
    const auto transfer = [&](std::uintptr_t packed, std::uintptr_t state,
                              std::uint64_t bytes) {
      copy(gather ? packed : state, bytes, gather ? state : packed);
    };
    transfer(packed_conv + row * conv_bytes,
             layout.conv_row_addresses[row], conv_bytes);
    transfer(packed_recurrent + row * recurrent_bytes,
             layout.recurrent_row_addresses[row], recurrent_bytes);
  }
}

}  // namespace statecentric
