# AgentX workload-fit audit — 2026-10-05

- Evidence label: `dataset-static-analysis/unqualified`
- Online performance claim permitted: **no**
- Dataset revision: `8fecd2fc56694469f758f0afbbb6335ad3043740`
- Trace SHA-256: `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`
- Analysis archive SHA-256: `2051056b89f4d03f70d8605c6151960f1ff338ff191cf1c8e53d49b0511f0058`

The 68,266 leaf requests have input p50 88,768 and p95 228,224 tokens, so chunked prefill work is structurally important. That does not establish the scheduler's exact dual-sequence, 64-token-per-sequence admission rate.

This V10 mechanism is not a per-token decode optimization. It requires exactly two sequences with 64 tokens each, the post-convolution pack path, and batched projection. Its historical full-tail online evidence regressed.

Decision: retain only as a third-line chunk/prefill candidate. Require exact 2×64 activation telemetry before retesting; do not infer eligibility from long output lengths or describe it as a decode gain.
