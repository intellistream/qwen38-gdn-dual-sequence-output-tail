#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != r210 ]]; then
  echo "usage: freeze_qwen38_tp2pp2_gdn_dual_sequence_tail.sh r210" >&2
  exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
revision=$1
base_revision=r153
base=/data/statecentric-builds/qwen38-tp2pp2-candidate-$base_revision
candidate=/data/statecentric-builds/qwen38-tp2pp2-candidate-$revision
provider=/data/statecentric-builds/qwen38-dual-sequence-tail-r210-provider-build/libstatecentric_qwen38_geometry_normalized_gdn_provider.so

base_manifest_sha=5e7a208b1a1891ebf294e1124a40c2eb602e4fde0504d9d0361abf22d0da1dae
plan_sha=18ed9e3ca676bc82d5fa874e9cbe5c9177453cd0919cbd345ffc8290cb99d45e
provider_sha=474dd9462c40e13ae15accfe19d6e57685a29a21eea346d25ae69af43c267459
r208b_stdout="$repo/results/qwen38-native-r208b-dual-sequence-tail-20260902-r208b-component/component.stdout"
r208b_stdout_sha=d316155147b4e85ba8c679e94dc8092f1ddb9af717e1d907264f1aff6441a827
r208b_manifest="$repo/results/qwen38-native-r208b-dual-sequence-tail-20260902-r208b-component/component/manifest.json"
r208b_manifest_sha=2a686f611d103e4afeb5b2bd4a0b1d9c2353fade18b9ab8ac0a183fc311266b3
r208b_evidence=0c409dc55e5903235deaf61daccc5181c2944d74d528f7e79830168f10da3fe7
r209_stdout="$repo/results/qwen38-native-r209-dual-sequence-tail-benchmark-20260902-r209-component/component.stdout"
r209_stdout_sha=414bf3b2483db44963454a0c4985ebf01ca5b0076b2624d358ae3f5dfcce413a
r209_manifest="$repo/results/qwen38-native-r209-dual-sequence-tail-benchmark-20260902-r209-component/component/manifest.json"
r209_manifest_sha=684f326ba737992e5fd5fcbb03da5baa7df04f5ac5f402505a4cfa2679562aba
r209_evidence=40a77e0ee9a2c3085105090b50db27ff5f833cea98ef0f906a3df3778143d774

[[ -d "$base" && ! -w "$base" ]] || {
  echo "immutable r153 base candidate is missing or writable: $base" >&2
  exit 2
}
[[ ! -e "$candidate" ]] || {
  echo "refusing to overwrite dual-sequence-tail candidate: $candidate" >&2
  exit 2
}
for binding in \
  "$base/candidate-manifest-r001.json:$base_manifest_sha" \
  "$base/optimization-plan.json:$plan_sha" \
  "$provider:$provider_sha" \
  "$r208b_stdout:$r208b_stdout_sha" \
  "$r208b_manifest:$r208b_manifest_sha" \
  "$r209_stdout:$r209_stdout_sha" \
  "$r209_manifest:$r209_manifest_sha"; do
  path=${binding%:*}
  expected=${binding##*:}
  [[ -f "$path" && $(sha256sum "$path" | cut -d' ' -f1) == "$expected" ]] || {
    echo "r210 source evidence binding failed: $path" >&2
    exit 3
  }
done

mkdir -p "$candidate/layouts"
find "$base" -maxdepth 1 -type f \
  ! -name 'libstatecentric_qwen38_gdn_provider.so' \
  -exec cp --preserve=mode,timestamps {} "$candidate/" \;
cp --preserve=mode,timestamps "$base"/layouts/rank-*.layout.tsv \
  "$candidate/layouts/"
install -m 0555 "$provider" "$candidate/libstatecentric_qwen38_gdn_provider.so"

jq --arg old "$base" --arg new "$candidate" --arg provider "$provider_sha" '
  walk(if type == "string" then gsub($old; $new) else . end) |
  .artifacts.gdn.sha256 = $provider
' "$base/candidate-manifest-r001.json" \
  >"$candidate/candidate-manifest-r001.json.tmp"
mv "$candidate/candidate-manifest-r001.json.tmp" \
  "$candidate/candidate-manifest-r001.json"

jq -n \
  --arg revision "$revision" \
  --arg base_revision "$base_revision" \
  --arg base_manifest "$base_manifest_sha" \
  --arg provider "$provider_sha" \
  --arg r208b_stdout "$r208b_stdout_sha" \
  --arg r208b_manifest "$r208b_manifest_sha" \
  --arg r208b_evidence "$r208b_evidence" \
  --arg r209_stdout "$r209_stdout_sha" \
  --arg r209_manifest "$r209_manifest_sha" \
  --arg r209_evidence "$r209_evidence" '
  {
    schema:"statecentric-qwen38-gdn-dual-sequence-tail-v1",
    revision:$revision,
    base_revision:$base_revision,
    base_manifest_sha256:$base_manifest,
    provider_sha256:$provider,
    enabled_environment:"STATECENTRIC_QWEN38_GDN_DUAL_SEQUENCE_TAIL_V10=1",
    admission:{
      requires_batched_projection_v1:true,
      recurrent_prefill_variant:5,
      state_count:2,
      tokens_per_sequence:[64,64],
      total_tokens:128,
      stateful_core_geometry:"per-sequence-M64",
      stateless_output_tail_geometry:"B2xM64"
    },
    fallback:"all-unadmitted-geometries-and-disabled-flag-use-the-r153-path",
    evidence:{
      exact_component_equivalence:{
        revision:"r208b",stdout_sha256:$r208b_stdout,
        manifest_sha256:$r208b_manifest,evidence_sha256:$r208b_evidence,
        output_elements:655360,mismatches:0,max_abs_error:0
      },
      component_timing:{
        revision:"r209",stdout_sha256:$r209_stdout,
        manifest_sha256:$r209_manifest,evidence_sha256:$r209_evidence,
        device_control_p50_us:194.005001336,
        device_candidate_p50_us:132.105000317,
        candidate_over_control_ratio_minus_one:0.468566677042,
        scope:"component-only; no online performance claim"
      }
    }
  }
' >"$candidate/gdn-dual-sequence-tail-v10.json"

while IFS=$'\t' read -r path expected; do
  [[ -f "$path" && $(sha256sum "$path" | cut -d' ' -f1) == "$expected" ]] || {
    echo "candidate artifact binding failed: $path" >&2
    exit 3
  }
done < <(jq -r '.artifacts | to_entries[] | [.value.path,.value.sha256] | @tsv' \
  "$candidate/candidate-manifest-r001.json")
[[ $(sha256sum "$candidate/optimization-plan.json" | cut -d' ' -f1) == \
   "$plan_sha" ]]

chmod 0555 "$candidate" "$candidate/layouts"
find "$candidate" -type f \( -name '*.json' -o -name '*.tsv' \) -print0 |
  xargs -0 chmod 0444
find "$candidate" -maxdepth 1 -type f ! -name '*.json' -exec chmod 0555 {} +
sha256sum "$candidate/candidate-manifest-r001.json" \
  "$candidate/optimization-plan.json" \
  "$candidate/libstatecentric_qwen38_gdn_provider.so" \
  "$candidate/gdn-dual-sequence-tail-v10.json"
