#pragma once

#include <cstdint>
#include <span>

namespace statecentric {

struct Qwen38GdnDualSequenceTailPlan {
  std::uint32_t batch = 0;
  std::uint32_t tokens_per_sequence = 0;
  bool batch_rms_norm = false;
  bool batch_elementwise = false;
  bool batch_output_projection = false;
};

struct Qwen38GdnPackedDecodePlan {
  std::uint32_t batch = 0;
  std::uint32_t tokens_per_sequence = 0;
  bool packed_recurrent = false;
};

struct Qwen38GdnLaneLocalPackedDecodePlan {
  std::uint32_t batch = 0;
  std::uint32_t tokens_per_sequence = 0;
  bool lane_prepare = false;
  bool packed_recurrent = false;
  bool lane_output_tail = false;
};

inline bool Qwen38GdnDualSequenceTailFeaturesValid(
    bool batched_projection_v1, bool dual_sequence_tail_v10) {
  return !dual_sequence_tail_v10 || batched_projection_v1;
}

inline bool Qwen38GdnProjectionOnlyTailFeaturesValid(
    bool batched_projection_v1, bool dual_sequence_tail_v10,
    bool projection_only_tail_v11) {
  return (!projection_only_tail_v11 || batched_projection_v1) &&
         !(dual_sequence_tail_v10 && projection_only_tail_v11);
}

inline bool Qwen38GdnPackedDecodeFeaturesValid(
    bool batched_projection_v1, bool dual_sequence_tail_v10,
    bool projection_only_tail_v11, bool packed_decode_recurrent_v12) {
  return !packed_decode_recurrent_v12 ||
         (batched_projection_v1 && !dual_sequence_tail_v10 &&
          !projection_only_tail_v11);
}

inline bool Qwen38GdnLaneLocalPackedDecodeFeaturesValid(
    bool batched_projection_v1, bool dual_sequence_tail_v10,
    bool projection_only_tail_v11, bool packed_decode_recurrent_v12,
    bool lane_local_packed_recurrent_v13) {
  return !lane_local_packed_recurrent_v13 ||
         (batched_projection_v1 && !dual_sequence_tail_v10 &&
          !projection_only_tail_v11 && !packed_decode_recurrent_v12);
}

inline bool Qwen38GdnLaneAnchoredPackedDecodeFeaturesValid(
    bool batched_projection_v1, bool dual_sequence_tail_v10,
    bool projection_only_tail_v11, bool packed_decode_recurrent_v12,
    bool lane_local_packed_recurrent_v13,
    bool lane_anchored_packed_recurrent_v14) {
  return !lane_anchored_packed_recurrent_v14 ||
         (batched_projection_v1 && !dual_sequence_tail_v10 &&
          !projection_only_tail_v11 && !packed_decode_recurrent_v12 &&
          !lane_local_packed_recurrent_v13);
}

inline bool Qwen38GdnPackedDecodeLayoutReady(bool post_conv_pack_v5,
                                              bool decode_forward_v1) {
  return post_conv_pack_v5 || decode_forward_v1;
}

inline bool Qwen38GdnPackedDecodeConstantsRequired(
    bool packed_decode_recurrent_v12, bool lane_local_packed_recurrent_v13,
    bool lane_anchored_packed_recurrent_v14) {
  return packed_decode_recurrent_v12 || lane_local_packed_recurrent_v13 ||
         lane_anchored_packed_recurrent_v14;
}

inline Qwen38GdnDualSequenceTailPlan PlanQwen38GdnDualSequenceTail(
    bool enabled, bool multistate, bool post_conv_pack_v5,
    bool batched_projection_v1, std::uint64_t state_count,
    std::uint64_t total_tokens,
    std::span<const std::uint32_t> token_counts) {
  if (!enabled || !multistate || !post_conv_pack_v5 ||
      !batched_projection_v1 || state_count != 2 || total_tokens != 128 ||
      token_counts.size() != state_count || token_counts[0] != 64 ||
      token_counts[1] != 64) {
    return {};
  }
  return {2, 64, true, true, true};
}

inline Qwen38GdnDualSequenceTailPlan PlanQwen38GdnProjectionOnlyTail(
    bool enabled, bool multistate, bool post_conv_pack_v5,
    bool batched_projection_v1, std::uint64_t state_count,
    std::uint64_t total_tokens,
    std::span<const std::uint32_t> token_counts) {
  if (!enabled || !multistate || !post_conv_pack_v5 ||
      !batched_projection_v1 || state_count != 2 || total_tokens != 128 ||
      token_counts.size() != state_count || token_counts[0] != 64 ||
      token_counts[1] != 64) {
    return {};
  }
  return {2, 64, false, false, true};
}

inline Qwen38GdnPackedDecodePlan PlanQwen38GdnPackedDecode(
    bool enabled, bool multistate, bool post_conv_pack_v5,
    bool batched_projection_v1, std::uint64_t state_count,
    std::uint64_t total_tokens,
    std::span<const std::uint32_t> token_counts) {
  if (!enabled || !multistate || !post_conv_pack_v5 ||
      !batched_projection_v1 || state_count != 2 || total_tokens != 2 ||
      token_counts.size() != state_count || token_counts[0] != 1 ||
      token_counts[1] != 1) {
    return {};
  }
  return {2, 1, true};
}

inline Qwen38GdnLaneLocalPackedDecodePlan
PlanQwen38GdnLaneLocalPackedDecode(
    bool enabled, bool multistate, bool post_conv_pack_v5,
    bool batched_projection_v1, std::uint64_t state_count,
    std::uint64_t total_tokens,
    std::span<const std::uint32_t> token_counts) {
  if (!enabled || !multistate || !post_conv_pack_v5 ||
      !batched_projection_v1 || state_count != 2 || total_tokens != 2 ||
      token_counts.size() != state_count || token_counts[0] != 1 ||
      token_counts[1] != 1) {
    return {};
  }
  return {2, 1, true, true, true};
}

inline Qwen38GdnLaneLocalPackedDecodePlan
PlanQwen38GdnLaneAnchoredPackedDecode(
    bool enabled, bool multistate, bool post_conv_pack_v5,
    bool batched_projection_v1, std::uint64_t state_count,
    std::uint64_t total_tokens,
    std::span<const std::uint32_t> token_counts) {
  return PlanQwen38GdnLaneLocalPackedDecode(
      enabled, multistate, post_conv_pack_v5, batched_projection_v1,
      state_count, total_tokens, token_counts);
}

}  // namespace statecentric
