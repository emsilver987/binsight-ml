"""Build a BinSight manifest that also includes every TrashNet image.

BinSight frames keep their material_class. TrashNet folders use
trashnet_label_map, then trashnet_label_overrides relabels the photos
that do not match that folder default. Images are not copied.
filepath is relative to image_root, which must be a parent of both datasets,
so train.py can keep doing image_root / filepath.

item_id is prefixed with the source. BinSight ids still group every frame of
one capture. TrashNet ids stay one image per item.

Usage (from binsight-ml/):
    python src/make_manifest_binsight_augmented.py --config configs/binsight_augmented.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv

from src.make_manifest_binsight import build_rows as build_binsight_rows
from src.make_manifest_binsight import read_item_log, scan_captures
from src.make_manifest_trashnet import build_rows as build_trashnet_rows
from src.util import load_rules, load_yaml, resolve, write_manifest

SOURCES = ("binsight", "trashnet")


def _under(root: Path, child: Path) -> str:
    try:
        return child.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{child} is not inside image_root {root}") from exc


def read_overrides(path: Path, class_names: set[str]) -> dict[str, str]:
    """filepath relative to the TrashNet root -> BinSight class.

    The folder map is the default. This file is only the images whose
    BinSight class is not the folder default.
    """
    if not path.is_file():
        raise FileNotFoundError(f"TrashNet label overrides not found: {path}")
    overrides = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = [column for column in ("filepath", "label") if column not in fields]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            filepath = (row.get("filepath") or "").strip()
            label = (row.get("label") or "").strip()
            if not filepath or not label:
                raise ValueError(f"{path}:{line_number} has an empty field")
            if label not in class_names:
                raise ValueError(f"{path}:{line_number} label {label!r} is not a BinSight class")
            if filepath in overrides:
                raise ValueError(f"{path}:{line_number} repeats {filepath}")
            overrides[filepath] = label
    return overrides


def apply_overrides(rows: list[dict], overrides: dict[str, str]) -> int:
    by_path = {row["filepath"]: row for row in rows}
    missing = sorted(set(overrides) - set(by_path))
    if missing:
        raise ValueError("override paths are not in TrashNet: " + ", ".join(missing[:8]))
    changed = 0
    for filepath, label in overrides.items():
        row = by_path[filepath]
        if row["label"] == label:
            raise ValueError(f"{filepath} override is already the folder label {label}")
        row["label"] = label
        changed += 1
    return changed


def retarget(rows: list[dict], image_root: Path, source_root: Path, source: str) -> list[dict]:
    prefix = _under(image_root, source_root)
    retargeted = []
    for row in rows:
        filepath = f"{prefix}/{row['filepath']}" if prefix else row["filepath"]
        image = image_root / filepath
        if not image.is_file():
            raise FileNotFoundError(f"manifest path does not exist: {image}")
        retargeted.append(
            {
                "filepath": filepath,
                "item_id": f"{source}:{row['item_id']}",
                "label": row["label"],
            }
        )
    return retargeted


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the BinSight+TrashNet training manifest.")
    parser.add_argument("--config", default="configs/binsight_augmented.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    class_names = list(config["classes"])
    rules = load_rules(config["rules"])
    missing_rules = [name for name in class_names if name not in rules["name_to_stream"]]
    if missing_rules:
        raise ValueError(f"config classes missing from rules.yaml: {missing_rules}")

    label_map = config["trashnet_label_map"]
    unknown_labels = sorted({label for label in label_map.values() if label not in class_names})
    if unknown_labels:
        raise ValueError(f"trashnet_label_map points outside config classes: {unknown_labels}")

    image_root = resolve(config["image_root"])
    binsight_root = resolve(config["binsight_image_root"])
    trashnet_root = resolve(config["trashnet_image_root"])
    item_log = resolve(config["binsight_item_log"])

    binsight_rows = build_binsight_rows(
        read_item_log(item_log),
        scan_captures(binsight_root),
        set(class_names),
    )
    trashnet_rows = build_trashnet_rows(trashnet_root, label_map)
    overrides = read_overrides(resolve(config["trashnet_label_overrides"]), set(class_names))
    n_overrides = apply_overrides(trashnet_rows, overrides)
    rows = retarget(binsight_rows, image_root, binsight_root, "binsight")
    rows.extend(retarget(trashnet_rows, image_root, trashnet_root, "trashnet"))

    labels_for: dict[str, str] = {}
    for row in rows:
        previous = labels_for.get(row["item_id"])
        if previous is None:
            labels_for[row["item_id"]] = row["label"]
        elif previous != row["label"]:
            raise ValueError(f"item_id {row['item_id']} has labels {previous} and {row['label']}")
    paths = [row["filepath"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate filepath in the combined manifest")

    write_manifest(config["manifest"], rows)

    frame_counts: dict[str, dict[str, int]] = {source: defaultdict(int) for source in SOURCES}
    item_ids_by_label: dict[str, dict[str, set[str]]] = {source: defaultdict(set) for source in SOURCES}
    for row in rows:
        source = row["item_id"].split(":", 1)[0]
        frame_counts[source][row["label"]] += 1
        item_ids_by_label[source][row["label"]].add(row["item_id"])

    print(f"wrote {len(rows)} frames to {resolve(config['manifest'])}")
    print(f"relabeled {n_overrides} TrashNet images off their folder default")
    print(f"image root: {image_root}")
    print("material class -> stream -> binsight items/frames + trashnet images")
    for name in class_names:
        stream = rules["name_to_stream"][name]
        binsight_items = len(item_ids_by_label["binsight"][name])
        binsight_frames = frame_counts["binsight"][name]
        trashnet_images = frame_counts["trashnet"][name]
        marker = "" if trashnet_images else "  (BinSight only)"
        print(
            f"  {name:22} {stream:10} "
            f"binsight {binsight_items:4d} items {binsight_frames:4d} frames  "
            f"trashnet {trashnet_images:4d}{marker}"
        )


if __name__ == "__main__":
    main()
