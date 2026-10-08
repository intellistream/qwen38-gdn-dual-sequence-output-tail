#include "statecentric/qwen38_gdn_dual_sequence_tail_policy.h"
#include "statecentric/qwen38_gdn_graph_event_policy.h"

#include <array>
#include <cstdint>

int main() {
  constexpr auto eager_fresh =
      statecentric::PlanQwen38GdnGraphEvents(false, false);
  constexpr auto eager_pooled =
      statecentric::PlanQwen38GdnGraphEvents(false, true);
  constexpr auto captured =
      statecentric::PlanQwen38GdnGraphEvents(true, true);
  if (eager_fresh.acquire_reusable_pool || eager_fresh.create_fresh_ex_sync ||
      !eager_pooled.acquire_reusable_pool ||
      eager_pooled.create_fresh_ex_sync || captured.acquire_reusable_pool ||
      !captured.create_fresh_ex_sync) {
    return 14;
  }
  if (!statecentric::Qwen38GdnGraphCapturedLaneEventsFeaturesValid(false,
                                                                   false) ||
      !statecentric::Qwen38GdnGraphCapturedLaneEventsFeaturesValid(true,
                                                                  false) ||
      !statecentric::Qwen38GdnGraphCapturedLaneEventsFeaturesValid(true,
                                                                  true) ||
      statecentric::Qwen38GdnGraphCapturedLaneEventsFeaturesValid(false,
                                                                 true)) {
    return 15;
  }
  constexpr auto single_follower =
      statecentric::PlanQwen38GdnSingleFollowerStreams(true, true, false, 2);
  if (!statecentric::Qwen38GdnGraphSingleFollowerFeaturesValid(true, true) ||
      statecentric::Qwen38GdnGraphSingleFollowerFeaturesValid(false, true) ||
      !statecentric::Qwen38GdnEagerSingleFollowerFeaturesValid(true, true) ||
      statecentric::Qwen38GdnEagerSingleFollowerFeaturesValid(false, true) ||
      !statecentric::Qwen38GdnDecodeFlatOutputProjectionFeaturesValid(true,
                                                                      true) ||
      statecentric::Qwen38GdnDecodeFlatOutputProjectionFeaturesValid(false,
                                                                     true) ||
      !statecentric::Qwen38GdnParentLocalDecodeTailFeaturesValid(true,
                                                                 true) ||
      !statecentric::Qwen38GdnParentLocalDecodeTailFeaturesValid(false,
                                                                 false) ||
      statecentric::Qwen38GdnParentLocalDecodeTailFeaturesValid(false,
                                                                true) ||
      !single_follower.anchor_on_parent_stream ||
      !single_follower.follower_on_auxiliary_stream ||
      single_follower.synchronization_events != 4 ||
      single_follower.stream_waits != 4 ||
      statecentric::PlanQwen38GdnSingleFollowerStreams(false, true, false, 2)
          .anchor_on_parent_stream ||
      !statecentric::PlanQwen38GdnSingleFollowerStreams(false, true, true, 2)
           .anchor_on_parent_stream ||
      statecentric::PlanQwen38GdnSingleFollowerStreams(true, true, true, 3)
          .anchor_on_parent_stream) {
    return 16;
  }
  if (!statecentric::Qwen38GdnDualSequenceTailFeaturesValid(false, false) ||
      !statecentric::Qwen38GdnDualSequenceTailFeaturesValid(true, false) ||
      !statecentric::Qwen38GdnDualSequenceTailFeaturesValid(true, true) ||
      statecentric::Qwen38GdnDualSequenceTailFeaturesValid(false, true)) {
    return 1;
  }
  if (!statecentric::Qwen38GdnProjectionOnlyTailFeaturesValid(false, false,
                                                               false) ||
      !statecentric::Qwen38GdnProjectionOnlyTailFeaturesValid(true, false,
                                                               true) ||
      statecentric::Qwen38GdnProjectionOnlyTailFeaturesValid(false, false,
                                                              true) ||
      statecentric::Qwen38GdnProjectionOnlyTailFeaturesValid(true, true,
                                                              true)) {
    return 4;
  }
  if (!statecentric::Qwen38GdnPackedDecodeFeaturesValid(false, false, false,
                                                         false) ||
      !statecentric::Qwen38GdnPackedDecodeFeaturesValid(true, false, false,
                                                        true) ||
      statecentric::Qwen38GdnPackedDecodeFeaturesValid(false, false, false,
                                                       true) ||
      statecentric::Qwen38GdnPackedDecodeFeaturesValid(true, true, false,
                                                       true) ||
      statecentric::Qwen38GdnPackedDecodeFeaturesValid(true, false, true,
                                                       true)) {
    return 6;
  }
  if (!statecentric::Qwen38GdnLaneLocalPackedDecodeFeaturesValid(
          false, false, false, false, false) ||
      !statecentric::Qwen38GdnLaneLocalPackedDecodeFeaturesValid(
          true, false, false, false, true) ||
      statecentric::Qwen38GdnLaneLocalPackedDecodeFeaturesValid(
          false, false, false, false, true) ||
      statecentric::Qwen38GdnLaneLocalPackedDecodeFeaturesValid(
          true, true, false, false, true) ||
      statecentric::Qwen38GdnLaneLocalPackedDecodeFeaturesValid(
          true, false, true, false, true) ||
      statecentric::Qwen38GdnLaneLocalPackedDecodeFeaturesValid(
          true, false, false, true, true)) {
    return 8;
  }
  if (!statecentric::Qwen38GdnLaneAnchoredPackedDecodeFeaturesValid(
          false, false, false, false, false, false) ||
      !statecentric::Qwen38GdnLaneAnchoredPackedDecodeFeaturesValid(
          true, false, false, false, false, true) ||
      statecentric::Qwen38GdnLaneAnchoredPackedDecodeFeaturesValid(
          false, false, false, false, false, true) ||
      statecentric::Qwen38GdnLaneAnchoredPackedDecodeFeaturesValid(
          true, true, false, false, false, true) ||
      statecentric::Qwen38GdnLaneAnchoredPackedDecodeFeaturesValid(
          true, false, true, false, false, true) ||
      statecentric::Qwen38GdnLaneAnchoredPackedDecodeFeaturesValid(
          true, false, false, true, false, true) ||
      statecentric::Qwen38GdnLaneAnchoredPackedDecodeFeaturesValid(
          true, false, false, false, true, true)) {
    return 10;
  }
  if (statecentric::Qwen38GdnPackedDecodeLayoutReady(false, false) ||
      !statecentric::Qwen38GdnPackedDecodeLayoutReady(true, false) ||
      !statecentric::Qwen38GdnPackedDecodeLayoutReady(false, true) ||
      !statecentric::Qwen38GdnPackedDecodeLayoutReady(true, true)) {
    return 12;
  }
  if (statecentric::Qwen38GdnPackedDecodeConstantsRequired(false, false,
                                                            false) ||
      !statecentric::Qwen38GdnPackedDecodeConstantsRequired(true, false,
                                                             false) ||
      !statecentric::Qwen38GdnPackedDecodeConstantsRequired(false, true,
                                                             false) ||
      !statecentric::Qwen38GdnPackedDecodeConstantsRequired(false, false,
                                                             true)) {
    return 13;
  }
  constexpr std::array<std::uint32_t, 2> exact{64, 64};
  const auto admitted = statecentric::PlanQwen38GdnDualSequenceTail(
      true, true, true, true, 2, 128, exact);
  if (admitted.batch != 2 || admitted.tokens_per_sequence != 64 ||
      !admitted.batch_rms_norm || !admitted.batch_elementwise ||
      !admitted.batch_output_projection) {
    return 2;
  }
  constexpr std::array<std::uint32_t, 2> ragged{63, 65};
  if (statecentric::PlanQwen38GdnDualSequenceTail(
          false, true, true, true, 2, 128, exact)
          .batch != 0 ||
      statecentric::PlanQwen38GdnDualSequenceTail(
          true, false, true, true, 2, 128, exact)
          .batch != 0 ||
      statecentric::PlanQwen38GdnDualSequenceTail(
          true, true, false, true, 2, 128, exact)
          .batch != 0 ||
      statecentric::PlanQwen38GdnDualSequenceTail(
          true, true, true, false, 2, 128, exact)
          .batch != 0 ||
      statecentric::PlanQwen38GdnDualSequenceTail(
          true, true, true, true, 2, 127, exact)
          .batch != 0 ||
      statecentric::PlanQwen38GdnDualSequenceTail(
          true, true, true, true, 2, 128, ragged)
          .batch != 0) {
    return 3;
  }
  const auto projection_only =
      statecentric::PlanQwen38GdnProjectionOnlyTail(
          true, true, true, true, 2, 128, exact);
  if (projection_only.batch != 2 ||
      projection_only.tokens_per_sequence != 64 ||
      projection_only.batch_rms_norm || projection_only.batch_elementwise ||
      !projection_only.batch_output_projection ||
      statecentric::PlanQwen38GdnProjectionOnlyTail(
          true, true, true, true, 2, 128, ragged)
          .batch != 0) {
    return 5;
  }
  constexpr std::array<std::uint32_t, 2> decode_exact{1, 1};
  constexpr std::array<std::uint32_t, 2> decode_ragged{1, 2};
  const auto packed_decode = statecentric::PlanQwen38GdnPackedDecode(
      true, true, true, true, 2, 2, decode_exact);
  if (packed_decode.batch != 2 || packed_decode.tokens_per_sequence != 1 ||
      !packed_decode.packed_recurrent ||
      statecentric::PlanQwen38GdnPackedDecode(
          false, true, true, true, 2, 2, decode_exact)
              .packed_recurrent ||
      statecentric::PlanQwen38GdnPackedDecode(
          true, false, true, true, 2, 2, decode_exact)
              .packed_recurrent ||
      statecentric::PlanQwen38GdnPackedDecode(
          true, true, false, true, 2, 2, decode_exact)
              .packed_recurrent ||
      statecentric::PlanQwen38GdnPackedDecode(
          true, true, true, false, 2, 2, decode_exact)
              .packed_recurrent ||
      statecentric::PlanQwen38GdnPackedDecode(
          true, true, true, true, 2, 3, decode_ragged)
              .packed_recurrent) {
    return 7;
  }
  const auto lane_local = statecentric::PlanQwen38GdnLaneLocalPackedDecode(
      true, true, true, true, 2, 2, decode_exact);
  if (lane_local.batch != 2 || lane_local.tokens_per_sequence != 1 ||
      !lane_local.lane_prepare || !lane_local.packed_recurrent ||
      !lane_local.lane_output_tail ||
      statecentric::PlanQwen38GdnLaneLocalPackedDecode(
          false, true, true, true, 2, 2, decode_exact)
          .packed_recurrent ||
      statecentric::PlanQwen38GdnLaneLocalPackedDecode(
          true, false, true, true, 2, 2, decode_exact)
          .packed_recurrent ||
      statecentric::PlanQwen38GdnLaneLocalPackedDecode(
          true, true, false, true, 2, 2, decode_exact)
          .packed_recurrent ||
      statecentric::PlanQwen38GdnLaneLocalPackedDecode(
          true, true, true, false, 2, 2, decode_exact)
          .packed_recurrent ||
      statecentric::PlanQwen38GdnLaneLocalPackedDecode(
          true, true, true, true, 2, 3, decode_ragged)
          .packed_recurrent) {
    return 9;
  }
  const auto lane_anchored =
      statecentric::PlanQwen38GdnLaneAnchoredPackedDecode(
          true, true, true, true, 2, 2, decode_exact);
  if (lane_anchored.batch != 2 ||
      lane_anchored.tokens_per_sequence != 1 ||
      !lane_anchored.lane_prepare || !lane_anchored.packed_recurrent ||
      !lane_anchored.lane_output_tail ||
      statecentric::PlanQwen38GdnLaneAnchoredPackedDecode(
          true, true, true, true, 2, 3, decode_ragged)
          .packed_recurrent) {
    return 11;
  }
  return 0;
}
