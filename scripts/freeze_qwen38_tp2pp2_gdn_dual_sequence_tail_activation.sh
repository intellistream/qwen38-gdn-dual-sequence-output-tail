#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != r215 ]]; then
  echo "usage: freeze_qwen38_tp2pp2_gdn_dual_sequence_tail_activation.sh r215" >&2
  exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
revision=$1
base_revision=r210
base=/data/statecentric-builds/qwen38-tp2pp2-candidate-$base_revision
candidate=/data/statecentric-builds/qwen38-tp2pp2-candidate-$revision
base_manifest_sha=df1e6e03d7f55e79c99a0218176ce8d9abd3e25549d859096ef884fbe4dd88d9
plan_sha=18ed9e3ca676bc82d5fa874e9cbe5c9177453cd0919cbd345ffc8290cb99d45e
provider_sha=474dd9462c40e13ae15accfe19d6e57685a29a21eea346d25ae69af43c267459
feature_contract_sha=52bec9162837fd6dc306097e117ad223be886e7de6dcc7f9162cd2b98208666e
worker_launcher="$repo/scripts/run_qwen38_tp2pp2_worker_container.sh"
worker_launcher_sha=49db5d9e6f33cbe00a7921ff0798faf09740a2e7ac1e4fbbedcf082434725854

[[ -d "$base" && ! -w "$base" ]] || {
  echo "immutable r210 base candidate is missing or writable: $base" >&2
  exit 2
}
[[ ! -e "$candidate" ]] || {
  echo "refusing to overwrite activation-fixed candidate: $candidate" >&2
  exit 2
}
for binding in \
  "$base/candidate-manifest-r001.json:$base_manifest_sha" \
  "$base/optimization-plan.json:$plan_sha" \
  "$base/libstatecentric_qwen38_gdn_provider.so:$provider_sha" \
  "$base/gdn-dual-sequence-tail-v10.json:$feature_contract_sha" \
  "$worker_launcher:$worker_launcher_sha"; do
  path=${binding%:*}
  expected=${binding##*:}
  [[ -f "$path" && $(sha256sum "$path" | cut -d' ' -f1) == "$expected" ]] || {
    echo "r215 source binding failed: $path" >&2
    exit 3
  }
done

mkdir -p "$candidate/layouts"
find "$base" -maxdepth 1 -type f \
  -exec cp --preserve=mode,timestamps {} "$candidate/" \;
cp --preserve=mode,timestamps "$base"/layouts/rank-*.layout.tsv \
  "$candidate/layouts/"

jq --arg old "$base" --arg new "$candidate" '
  walk(if type == "string" then gsub($old; $new) else . end)
' "$base/candidate-manifest-r001.json" \
  >"$candidate/candidate-manifest-r001.json.tmp"
mv "$candidate/candidate-manifest-r001.json.tmp" \
  "$candidate/candidate-manifest-r001.json"

jq -n \
  --arg revision "$revision" \
  --arg base_revision "$base_revision" \
  --arg base_manifest "$base_manifest_sha" \
  --arg feature_contract "$feature_contract_sha" \
  --arg worker_launcher "$worker_launcher_sha" '
  {
    schema:"statecentric-qwen38-gdn-dual-sequence-tail-activation-v1",
    revision:$revision,
    base_revision:$base_revision,
    base_manifest_sha256:$base_manifest,
    feature_contract_sha256:$feature_contract,
    worker_launcher_sha256:$worker_launcher,
    activation_chain:[
      "native service exports STATECENTRIC_QWEN38_GDN_DUAL_SEQUENCE_TAIL_V10=1",
      "worker wrapper validates V10 is binary and requires V1",
      "docker environment explicitly forwards V10",
      "provider admits only exact B2xM64 recurrent-prefill-v5 geometry"
    ],
    supersedes_online_evidence:["r211","r212","r213b","r214"],
    supersession_reason:"r210 host export was not forwarded into the worker container, so the provider executed its V10-off fallback",
    performance_claim:false
  }
' >"$candidate/gdn-dual-sequence-tail-activation-v10.json"

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
  "$candidate/gdn-dual-sequence-tail-v10.json" \
  "$candidate/gdn-dual-sequence-tail-activation-v10.json"
