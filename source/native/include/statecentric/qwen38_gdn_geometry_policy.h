#ifndef STATECENTRIC_QWEN38_GDN_GEOMETRY_POLICY_H_
#define STATECENTRIC_QWEN38_GDN_GEOMETRY_POLICY_H_

#include <cstdint>

namespace statecentric {

// Normalize the recurrence algorithm across scheduler geometries without
// changing the common packed projection, convolution, or gating geometry.
constexpr bool UsePackedSequenceChunkGdn(
    bool multistate, bool prefill_post_conv_pack_v5,
    bool recurrent_prefill_always_recurrent, std::uint32_t sequence_tokens) {
  return multistate && prefill_post_conv_pack_v5 &&
         !recurrent_prefill_always_recurrent && sequence_tokens >= 9 &&
         sequence_tokens <= 64;
}

}  // namespace statecentric

#endif  // STATECENTRIC_QWEN38_GDN_GEOMETRY_POLICY_H_
