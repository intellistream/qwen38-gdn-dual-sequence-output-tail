#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace statecentric {

struct Qwen38GdnSequenceStreamPlan {
  std::uint32_t sequence_count = 0;
  std::array<std::uint32_t, 4> lane_by_sequence{};
};

inline std::optional<Qwen38GdnSequenceStreamPlan>
PlanQwen38GdnSequenceStreams(std::uint64_t sequence_count) {
  if (sequence_count == 0 || sequence_count > 4) return std::nullopt;
  Qwen38GdnSequenceStreamPlan plan{};
  plan.sequence_count = static_cast<std::uint32_t>(sequence_count);
  for (std::uint32_t sequence = 0; sequence < plan.sequence_count; ++sequence) {
    plan.lane_by_sequence[sequence] = sequence;
  }
  return plan;
}

}  // namespace statecentric
