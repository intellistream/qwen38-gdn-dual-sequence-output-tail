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
    assert metadata["maintainers"] == ["ShuhaoZhangTony"]
    assert metadata["advisor_status"] == "none"
    assert metadata["advisors"] == []
    assert metadata["lifecycle_status"] == "deprecated"
    assert metadata["website_visibility"] == "excluded"
