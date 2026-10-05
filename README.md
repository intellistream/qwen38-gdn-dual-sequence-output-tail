# Qwen3.8 dual-sequence GDN output-tail batching (corrected)

Corrects the V10 activation record: r210 did not forward the feature flag into workers, while activation-fixed r215 batches the full stateless output tail only as an explicitly deprecated experiment.

## Status

This is an independent, evidence-bound native mod repository. It is disabled by default and does not inherit a performance qualification. Component gains and online regressions in `versions/*/mod.json` must both be retained.

The proposal author is unverified: `Qwen3.8` identifies the target model family, not a proven proposing agent.

## Contents

- Immutable version manifests and SHA-256 catalog bindings.
- The pinned native provider and policy-header source snapshot from StateAxis commit `59c4e51315df2fce0c8978820b1e51c2bf78e169`.
- Mechanism-specific freeze and telemetry scripts.
- Validation tests that reject inherited qualification or invented proposal provenance.
- Evidence notes retain both the historical online regression and newer bounded
  component screens; see
  [`evidence/2026-10-05-dual-tail-flat-output.md`](evidence/2026-10-05-dual-tail-flat-output.md).

## Validate

```bash
python -m pip install -e '.[test]'
python tools/validate_repository.py
pytest -q
```

## Activation boundary

Activation remains the manual engine/artifact route recorded in the manifest. Do not enable this mod in production or reinterpret component measurements as online gains.

The newer flattened final projection refines the exact 2 x 64 chunk/prefill
component only. It does not change the mod's deprecated, disabled-by-default
status. Promotion still requires matched TP2 x PP2 AgentX online evidence.
