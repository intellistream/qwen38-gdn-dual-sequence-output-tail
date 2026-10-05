#pragma once

#include <cstdint>
#include <span>

namespace statecentric {

struct Qwen38GdnBatchedProjectionPlan {
  std::uint32_t batch = 0;
  std::uint32_t tokens_per_sequence = 0;
  bool qkv = false;
  bool z = false;
  bool a = false;
  bool b = false;
  bool parallel_qkv_z = false;
};

inline Qwen38GdnBatchedProjectionPlan PlanQwen38GdnBatchedProjection(
    bool enabled, bool parallel_enabled, std::uint64_t state_count,
    std::uint64_t total_tokens,
    std::span<const std::uint32_t> token_counts) {
  if (!enabled || state_count != 2 || total_tokens != 128 ||
      token_counts.size() != state_count || token_counts[0] != 64 ||
      token_counts[1] != 64) {
    return {};
  }
  return {2, 64, true, true, false, false, parallel_enabled};
}

}  // namespace statecentric
