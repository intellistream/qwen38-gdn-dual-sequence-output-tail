# AgentX workload-fit audit — 2026-10-05

- Evidence label: `dataset-static-analysis/unqualified`
- Online performance claim permitted: **no**
- Dataset revision: `8fecd2fc56694469f758f0afbbb6335ad3043740`
- Trace SHA-256: `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`
- Analysis archive SHA-256: `2051056b89f4d03f70d8605c6151960f1ff338ff191cf1c8e53d49b0511f0058`

The 68,266 leaf requests have output p50 376 and p95 3,263 tokens; 98.45% of output-token exposure comes from requests producing at least 128 tokens. A repeated stateless output tail is therefore relevant, but this mechanism admits only exact dual-sequence geometry.

Decision: retain as a third-line candidate. Test only after lane anchoring and projection-only batching because historical full-tail online evidence regressed and the corpus does not prove the runtime admission rate.

