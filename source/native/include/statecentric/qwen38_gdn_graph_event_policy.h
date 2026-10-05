#pragma once

namespace statecentric {

struct Qwen38GdnGraphEventPlan {
  bool acquire_reusable_pool = false;
  bool create_fresh_ex_sync = false;
};

struct Qwen38GdnCapturedLaneStreamPlan {
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

// The original two-auxiliary-lane graph needs five events and six waits per
// GDN layer.  During model-RI replay those waits expand into CAPTURE_WAIT
// tasks.  Keeping lane 0 on the parent stream removes its fork and join while
// preserving lane 1 overlap around the packed recurrent midpoint.
inline constexpr Qwen38GdnCapturedLaneStreamPlan
PlanQwen38GdnCapturedLaneStreams(bool model_ri_capture_active,
                                bool graph_single_follower_v16,
                                unsigned state_count) noexcept {
  if (model_ri_capture_active && graph_single_follower_v16 &&
      state_count == 2) {
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
