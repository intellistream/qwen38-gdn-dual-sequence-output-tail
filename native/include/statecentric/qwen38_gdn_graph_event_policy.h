#pragma once

namespace statecentric {

struct Qwen38GdnGraphEventPlan {
  bool acquire_reusable_pool = false;
  bool create_fresh_ex_sync = false;
};

struct Qwen38GdnSingleFollowerStreamPlan {
  bool anchor_on_parent_stream = false;
  bool follower_on_auxiliary_stream = false;
  unsigned synchronization_events = 0;
  unsigned stream_waits = 0;
};

inline constexpr bool Qwen38GdnGraphCapturedLaneEventsFeaturesValid(
    bool lane_anchored_packed_recurrent_v14,
    bool graph_captured_lane_events_v15) noexcept {
  return !graph_captured_lane_events_v15 ||
         lane_anchored_packed_recurrent_v14;
}

inline constexpr bool Qwen38GdnGraphSingleFollowerFeaturesValid(
    bool graph_captured_lane_events_v15,
    bool graph_single_follower_v16) noexcept {
  return !graph_single_follower_v16 || graph_captured_lane_events_v15;
}

inline constexpr bool Qwen38GdnEagerSingleFollowerFeaturesValid(
    bool graph_single_follower_v16,
    bool eager_single_follower_v17) noexcept {
  return !eager_single_follower_v17 || graph_single_follower_v16;
}

inline constexpr bool Qwen38GdnDecodeFlatOutputProjectionFeaturesValid(
    bool eager_single_follower_v17,
    bool decode_flat_output_projection_v18) noexcept {
  return !decode_flat_output_projection_v18 || eager_single_follower_v17;
}

inline constexpr bool Qwen38GdnParentLocalDecodeTailFeaturesValid(
    bool decode_flat_output_projection_v18,
    bool parent_local_decode_tail_v19) noexcept {
  return !parent_local_decode_tail_v19 || decode_flat_output_projection_v18;
}

// The original two-auxiliary-lane path needs five events and six waits per GDN
// layer. Keeping lane 0 on the parent stream removes its fork and join while
// preserving lane 1 overlap around the packed recurrent midpoint. V16 applies
// this topology to model-RI capture; V17 extends it to eager execution.
inline constexpr Qwen38GdnSingleFollowerStreamPlan
PlanQwen38GdnSingleFollowerStreams(bool model_ri_capture_active,
                                  bool graph_single_follower_v16,
                                  bool eager_single_follower_v17,
                                  unsigned state_count) noexcept {
  const bool captured_single_follower =
      model_ri_capture_active && graph_single_follower_v16;
  const bool eager_single_follower =
      !model_ri_capture_active && eager_single_follower_v17;
  if ((captured_single_follower || eager_single_follower) && state_count == 2) {
    return {true, true, 4, 4};
  }
  return {};
}

// model-RI owns every stream/event edge in a captured cross-stream graph.
// A graph therefore needs graph-lifetime Ex-SYNC events and cannot borrow an
// event-pool slot whose lifetime and reuse are governed by eager submissions.
inline constexpr Qwen38GdnGraphEventPlan PlanQwen38GdnGraphEvents(
    bool model_ri_capture_active, bool reusable_pool_enabled) noexcept {
  if (model_ri_capture_active) return {false, true};
  return {reusable_pool_enabled, false};
}

}  // namespace statecentric
