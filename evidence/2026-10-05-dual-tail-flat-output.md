# Dual-tail flattened-output refinement — 2026-10-05

## Evidence boundary

- Label: `component-only/unqualified`; online claim: **not permitted**.
- Geometry: two sequences, 64 tokens per sequence, Qwen3.8 GDN stateless
  output tail; this is chunk/prefill work, not decode-one-token work.
- StateAxis source state: parent
  `936eedc5b8a4a9e113af5118656294d342d2e54c` plus the two-path patch archived
  as SHA-256
  `78c4c5201a1ea340f1cc80389de741d81486dd39a8b3988f5fd7089d2358f7ea`.
  The same source state is represented locally by commit
  `1498f464c1c1e26412ccdf8b186857e4a0c06ca7`; a canonical StateAxis merge
  commit must be recorded separately after review.
- Hardware/runtime: two visible Ascend 910B2 devices, CANN 9.1.0, Release
  build. Both devices ran concurrently; each device ran five independent
  process invocations with three warmups and 12 samples per arm.
- Model weights and AgentX service were not loaded. No submodule revision was
  involved.

The refinement preserves the existing V10 RMSNorm, SiLU and multiply path but
flattens the final `[2, 64, value]` gated tensor to a single M=128 matmul. The
control uses the existing shared-weight batch matmul.

## Raw evidence

The repository includes
[`raw/dual-tail-flat-output-20261005.jsonl.tar.gz`](raw/dual-tail-flat-output-20261005.jsonl.tar.gz),
SHA-256
`47a07203654d96b76685b2fa14def0e98e74d432a2a6017f179956f4195e3183`.

Archive members and SHA-256 values:

- `qwen38-dual-tail-pressure-device0.jsonl`:
  `1d314c445be9a6f528a4bd834b25afb48c1eebb5e7a1da85465c857cb95739cc`
- `qwen38-dual-tail-pressure-device1.jsonl`:
  `5a0e784f6cb30edb177383acc3be291157c9768566d0a243acdb59a4dc6668da`
- `qwen38-dual-tail-flat-device0.jsonl`:
  `49d1ab24fdbb758b67ea80114d24d24b518e4c22f437e83a55ed6cbe30827795`
- `qwen38-dual-tail-flat-device1.jsonl`:
  `dab29e39ed27274c06f6d9567439c6419600dfbd224c37d185b67e74f0408079`

## Results

All ten concurrent invocations were bitwise exact for both the old batch
projection and flattened projection.

| Comparison | Device p50 | Host p50 | Direction |
| --- | ---: | ---: | --- |
| Existing V10 full tail vs sequential | +52.416% median (43.523% to 59.400%) | +39.844% median (36.429% to 44.864%) | 10/10 positive on both |
| Flattened projection vs existing V10 projection | +1.958% median (0.303% to 5.578%) | +1.268% median (-0.214% to 4.888%) | device 10/10 positive; host 8/10 positive |
| Flattened full tail vs sequential | +55.517% median | +44.819% median | component only |

The incremental device gain is small but consistent. Host movement is marginal
and noisy, so it is not evidence of a service-level benefit.

A post-rebuild logical-device-0 smoke was also bitwise exact. It showed only a
0.649% device-p50 improvement and a 3.163% host-p50 regression for flattened
versus batch projection. Its raw JSONL has SHA-256
`7673d076b051c45915bef6dafef637c6ee131bb86f0b87d0493117927b1208d6`.
This single run is kept separate from the ten-run aggregate and strengthens the
decision not to promote on host timing.

## Validation and negative evidence

- Six focused provider, loader, geometry, event-pool and dual-tail policy tests
  passed (6/6).
- The provider and benchmark targets built successfully before the run.
- Full geometry-provider admission initially stopped because the installed
  custom-op library did not export `aclnnFusedGdnGatingGetWorkspaceSize`.
  Building the repository-pinned fused-GDN host API in a temporary workspace
  and preloading it with CANN `libopapi` resolved that packaging gap. The V10
  plus batched-projection geometry provider then admitted successfully and all
  six focused tests passed. The temporary shim SHA-256 was
  `987d3fccc8d25a6169477896d71d32068c65f058d18ad983a32f3385ed75c05c`;
  no installed custom-op file was modified.
- Activation-fixed historical V10 online evidence regressed wall time,
  throughput, TTFT, TPOT and E2E means. This component result does not erase or
  reinterpret that negative result.

## Decision

Retain the flattening change as a bounded implementation refinement for exact
2 x 64 chunk/prefill geometry. Keep the mod deprecated, disabled by default and
unqualified. Do not claim decode, AgentX, or online benefit. The promotion gate
remains a matched, repeatable four-device TP2 x PP2 AgentX run with activation
telemetry proving that the intended route executed.
