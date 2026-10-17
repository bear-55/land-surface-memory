"""Portable paths, frozen split identifiers, and file provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 29, 47, 65, 83, 103, 119, 137, 155, 173)
MODELS = ("B0", "A1", "M1")


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def config():
    return read_json(ROOT / "configs/formal.json")


def paths():
    local = ROOT / "configs/paths.local.json"
    result = read_json(local) if local.exists() else {}
    if os.environ.get("CAMELS_ROOT"):
        result["camels_root"] = str(Path(os.environ["CAMELS_ROOT"]).expanduser().resolve())
    return result


def split_ids():
    result = {role: (ROOT / f"splits/formal_pub_527/{role}.txt").read_text(
        encoding="utf-8-sig").split() for role in ("train", "val", "test")}
    for role, count in (("train", 370), ("val", 50), ("test", 107)):
        if len(result[role]) != count or any(len(x) != 8 or not x.isdigit() for x in result[role]):
            raise ValueError(f"Invalid frozen {role} basin list")
    if len(set(sum(result.values(), []))) != 527:
        raise ValueError("Frozen basin splits contain duplicates or overlap")
    return result


def selected_roster(models, seeds):
    names = list(MODELS) if models == ["all"] else list(models)
    numbers = list(SEEDS) if seeds == ["all"] else [int(s) for s in seeds]
    if not names or len(set(names)) != len(names) or not set(names).issubset(MODELS):
        raise ValueError("Choose unique models from B0, A1, and M1, or all")
    if not numbers or len(set(numbers)) != len(numbers) or not set(numbers).issubset(SEEDS):
        raise ValueError("Choose unique seeds from the published ten-seed roster, or all")
    return names, numbers
