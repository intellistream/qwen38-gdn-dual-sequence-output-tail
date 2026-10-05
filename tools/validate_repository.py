#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    schema = load(root / "schema/mod.schema.json")
    catalog = load(root / "catalog.json")
    provenance = load(root / "PROVENANCE.json")
    entries = catalog.get("mods", [])
    identities = {entry.get("mod_id") for entry in entries}
    if len(identities) != 1:
        errors.append("catalog must contain exactly one mod ID")
    if provenance.get("qualification_inherited") is not False:
        errors.append("repository must not inherit performance qualification")
    if provenance.get("proposal_origin", {}).get("classification") != "unverified":
        errors.append("proposal origin must remain unverified without primary evidence")
    source = root / provenance["source_snapshot"]["path"]
    if not source.is_file() or digest(source) != provenance["source_snapshot"]["sha256"]:
        errors.append("provider source snapshot digest mismatch")
    for entry in entries:
        path = root / entry["manifest"]
        if not path.is_file():
            errors.append(f"missing manifest: {entry['manifest']}")
            continue
        if digest(path) != entry["sha256"]:
            errors.append(f"manifest digest mismatch: {entry['manifest']}")
        document = load(path)
        for error in jsonschema.Draft202012Validator(schema).iter_errors(document):
            errors.append(f"{entry['manifest']}: {error.message}")
        if document.get("mod_id") != entry.get("mod_id"):
            errors.append(f"manifest mod ID mismatch: {entry['manifest']}")
        if document.get("version") != entry.get("version"):
            errors.append(f"manifest version mismatch: {entry['manifest']}")
        if document.get("default_enabled") is not False:
            errors.append(f"mod must remain disabled by default: {entry['manifest']}")
    return errors


if __name__ == "__main__":
    failures = validate()
    for failure in failures:
        print(f"ERROR: {failure}")
    raise SystemExit(bool(failures))
