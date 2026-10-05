# Dual-sequence output-tail component screen — 2026-10-05

## Evidence boundary

- Label: `component-only/unqualified`; online claim: **not permitted**.
- StateAxis source: `59c4e51315df2fce0c8978820b1e51c2bf78e169`, clean isolated worktree.
- Hardware/runtime: Ascend 910B2 logical device 0; two cards visible and idle before run; CANN 9.1.0; Release build.
- Model/graph: no weights loaded; synthetic batch 2 × 64-token Qwen3.8 GDN tail geometry; eager native component.
- Submodule: not applicable.
- Commands: `statecentric_qwen38_dual_sequence_tail_equivalence_probe 0` and `statecentric_qwen38_dual_sequence_tail_benchmark 0`.

## Results

Normalized, activated, gated, and projected outputs were bitwise exact. The benchmark used 3 warmups and 12 samples per arm.

- Sequential control device p50: 229.385 us.
- Batched candidate device p50: 155.375 us.
- Device p50 improvement: 47.63%.
- Host p50 improvement: 42.55%.
- Both batched measurements beat the bracketed control average; the component integration gate passed.

## Negative and limiting evidence

The final sequential arm contained an approximately 36.6 ms outlier. This test covers a 64-token tail, not decode-one-token or online scheduler behavior.

## Decision

Retain as the strongest prefill/chunk-tail component candidate. Next gate should target paired active sequences with equal 64-token tail geometry; disable the mod outside that bounded shape.
