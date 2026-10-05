from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validator", ROOT / "tools/validate_repository.py")
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_repository_is_valid() -> None:
    assert VALIDATOR.validate(ROOT) == []


def test_qualification_cannot_be_inherited(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    path = root / "PROVENANCE.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["qualification_inherited"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "repository must not inherit performance qualification" in VALIDATOR.validate(root)


def test_proposal_origin_cannot_be_invented(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    path = root / "PROVENANCE.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["proposal_origin"]["classification"] = "model-authored"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "proposal origin must remain unverified without primary evidence" in VALIDATOR.validate(root)
