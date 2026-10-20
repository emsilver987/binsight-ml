"""Paths, config, rules, manifests, and the item-level split.

Nothing in here imports TensorFlow. Manifest creation and the split stay
usable before a training environment exists.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_COLUMNS = ("filepath", "item_id", "label")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_yaml(path: str | Path) -> dict:
    with resolve(path).open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return data


def load_rules(path: str | Path) -> dict:
    raw = load_yaml(path)
    name_to_stream = {}
    name_to_id = {}
    classes = []
    for entry in raw["classes"]:
        name = entry["name"]
        classes.append(name)
        name_to_stream[name] = entry["stream"]
        name_to_id[name] = entry["id"]
    streams = [name for name in raw["streams"] if name != "Unknown"]
    return {
        "version": raw.get("version"),
        "jurisdiction": raw.get("jurisdiction"),
        "classes": classes,
        "name_to_stream": name_to_stream,
        "name_to_id": name_to_id,
        "streams": streams,
    }


def read_manifest(path: str | Path) -> list[dict]:
    path = resolve(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = [column for column in MANIFEST_COLUMNS if column not in fields]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            parsed = {column: (row.get(column) or "").strip() for column in MANIFEST_COLUMNS}
            if not all(parsed.values()):
                raise ValueError(f"{path}:{line_number} has an empty manifest field")
            rows.append(parsed)
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def write_manifest(path: str | Path, rows: list[dict]) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["filepath"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def group_items(rows: list[dict]) -> dict[str, dict]:
    """Group manifest rows by item_id. Every row of an item shares one label."""
    items: dict[str, dict] = {}
    for row in rows:
        item_id = row["item_id"]
        bucket = items.get(item_id)
        if bucket is None:
            items[item_id] = {"label": row["label"], "rows": [row]}
            continue
        if bucket["label"] != row["label"]:
            raise ValueError(
                f"item_id {item_id!r} has labels {bucket['label']!r} and {row['label']!r}"
            )
        bucket["rows"].append(row)
    return items


def limit_items_per_class(items: dict[str, dict], per_class: int, seed: int) -> dict[str, dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    for item_id, info in items.items():
        by_label[info["label"]].append(item_id)
    kept = {}
    for label in sorted(by_label):
        ids = by_label[label]
        rng.shuffle(ids)
        for item_id in ids[:per_class]:
            kept[item_id] = items[item_id]
    return kept


def split_item_ids(
    items: dict[str, dict],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    """Stratified item split. All rows of an item stay in one split.

    TrashNet has one image per item_id, so this becomes a stratified image
    split. BinSight item ids cover every frame of one physical object, and
    the same function keeps those frames together.
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to < 1")

    rng = random.Random(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    for item_id, info in items.items():
        by_label[info["label"]].append(item_id)

    splits = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        ids = by_label[label]
        rng.shuffle(ids)
        n_test, n_val = _split_counts(len(ids), val_fraction, test_fraction)
        splits["test"].extend(ids[:n_test])
        splits["val"].extend(ids[n_test : n_test + n_val])
        splits["train"].extend(ids[n_test + n_val :])
    return splits


def rows_for_split(items: dict[str, dict], item_ids: list[str]) -> list[dict]:
    rows = []
    for item_id in item_ids:
        rows.extend(items[item_id]["rows"])
    return rows


def count_by_label(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["label"]] += 1
    return dict(counts)


def _split_counts(n: int, val_fraction: float, test_fraction: float) -> tuple[int, int]:
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    while n_test + n_val > n:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break
    return n_test, n_val
