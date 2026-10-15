import json
from pathlib import Path

from vllm_hust_ext.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_archived_mod_has_manifest03_carrier_and_direct_owner() -> None:
    descriptor = load_manifest(
        ROOT
        / "src/qwen38_gdn_dual_sequence_output_tail/vllm-hust-extension-v0.3.json"
    )
    assert descriptor.schema_version in {"0.3", "0.3-experimental"}
    assert descriptor.bundle_id == (
        "org.vllm-hust.qwen38-gdn-dual-sequence-output-tail"
    )
    assert descriptor.resource_claims[0].mode == "exclusive"

    metadata = json.loads((ROOT / "MOD_METADATA.json").read_text())
    assert metadata["mod_id"] == descriptor.bundle_id
    responsibility = metadata["responsibility"]
    assert [item["github"] for item in responsibility["maintainers"]] == [
        "ShuhaoZhangTony"
    ]
    assert responsibility["advisor_status"] == "none"
    assert responsibility["advisors"] == []
    assert metadata["lifecycle"]["status"] == "deprecated"
    assert metadata["lifecycle"]["default_enabled"] is False
    assert "public MOD-page listing" in metadata["scope"]["out_of_scope"]
    assert metadata["evidence"]["qualification"] == "deprecated-unqualified"
