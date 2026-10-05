#pragma once

#include <array>
#include <cstddef>
#include <optional>

namespace statecentric {

constexpr std::size_t kQwen38SynchronizationEventBundleCount = 4;

inline std::optional<std::size_t> SelectQwen38SynchronizationEventBundle(
    const std::array<bool, kQwen38SynchronizationEventBundleCount>& in_use,
    std::size_t next_cursor,
    std::size_t active_bundle_count =
        kQwen38SynchronizationEventBundleCount) {
  if (active_bundle_count == 0 || active_bundle_count > in_use.size()) {
    return std::nullopt;
  }
  for (std::size_t offset = 0; offset < active_bundle_count; ++offset) {
    const auto index = (next_cursor + offset) % active_bundle_count;
    if (!in_use[index]) return index;
  }
  return std::nullopt;
}

}  // namespace statecentric
