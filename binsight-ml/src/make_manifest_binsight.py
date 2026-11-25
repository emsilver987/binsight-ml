"""Build the BinSight training manifest from the item log and image folders.

item_manifest.csv is the label source: one row per capture, with item_id
and material_class. The trainer never reads that file. It reads
filepath,item_id,label rows, one per frame. Images live at

    <image_root>/<material_class>/item<item_id>/*.{jpg,jpeg,png,bmp,webp}

Re-run this after adding items or frames. It rewrites the training manifest
from scratch and refuses to write when the log and the folders disagree:
a missing folder, an extra folder, a class-directory that does not match
material_class, or a capture with no images.

Usage (from binsight-ml/):
    python src/make_manifest_binsight.py --config configs/binsight.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.util import IMAGE_SUFFIXES, load_rules, load_yaml, resolve, write_manifest

LOG_COLUMNS = ("item_id", "material_class")
ITEM_DIR_PREFIX = "item"


def read_item_log(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"item log not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = [column for column in LOG_COLUMNS if column not in fields]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        rows = []
        seen = set()
        for line_number, row in enumerate(reader, start=2):
            item_id = (row.get("item_id") or "").strip()
            label = (row.get("material_class") or "").strip()
            if not item_id and not label:
                continue
            if not item_id or not label:
                raise ValueError(f"{path}:{line_number} is missing item_id or material_class")
            if item_id in seen:
                raise ValueError(f"{path}:{line_number} repeats item_id {item_id}")
            seen.add(item_id)
            rows.append({"item_id": item_id, "label": label})
    if not rows:
        raise ValueError(f"{path} has no item rows")
    return rows


def scan_captures(image_root: Path) -> dict[str, dict]:
    """Map item_id to the class directory and the frame paths under it.

    item_id is the folder name with the leading "item" removed, so item0001
    lines up with item_id 0001 in the log.
    """
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root not found: {image_root}")

    captures: dict[str, dict] = {}
    for class_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        for item_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            if not item_dir.name.startswith(ITEM_DIR_PREFIX):
                raise ValueError(
                    f"{item_dir.relative_to(image_root).as_posix()} "
                    f"is not named {ITEM_DIR_PREFIX}<item_id>"
                )
            item_id = item_dir.name[len(ITEM_DIR_PREFIX) :]
            if not item_id:
                raise ValueError(f"{item_dir} has no item_id after {ITEM_DIR_PREFIX!r}")
            frames = [
                path.relative_to(image_root).as_posix()
                for path in sorted(item_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
            if item_id in captures:
                previous = captures[item_id]["class_dir"]
                raise ValueError(
                    f"item_id {item_id} is in both {previous}/ and {class_dir.name}/"
                )
            captures[item_id] = {"class_dir": class_dir.name, "frames": frames}
    return captures


def build_rows(log_rows: list[dict], captures: dict[str, dict], class_names: set[str]) -> list[dict]:
    problems = []
    unknown = sorted({row["label"] for row in log_rows} - class_names)
    if unknown:
        problems.append(f"material_class values missing from config classes: {unknown}")

    logged = {row["item_id"]: row["label"] for row in log_rows}
    missing = sorted(set(logged) - set(captures))
    extra = sorted(set(captures) - set(logged))
    if missing:
        problems.append("item log rows with no image folder: " + ", ".join(missing))
    if extra:
        problems.append("image folders with no item log row: " + ", ".join(extra))

    mismatched = []
    empty = []
    for item_id, label in logged.items():
        capture = captures.get(item_id)
        if capture is None:
            continue
        if capture["class_dir"] != label:
            mismatched.append(f"{item_id} logged as {label} but stored under {capture['class_dir']}/")
        if not capture["frames"]:
            empty.append(item_id)
    if mismatched:
        problems.append("class folder does not match material_class:\n  " + "\n  ".join(mismatched))
    if empty:
        problems.append("captures with no images: " + ", ".join(empty))
    if problems:
        raise ValueError("item log and image folders disagree:\n" + "\n".join(problems))

    rows = []
    for item_id, label in logged.items():
        for filepath in captures[item_id]["frames"]:
            rows.append({"filepath": filepath, "item_id": item_id, "label": label})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the BinSight training manifest from the item log.")
    parser.add_argument("--config", default="configs/binsight.yaml")
    parser.add_argument(
        "--item-log",
        default=None,
        help="Override config item_log. Defaults to the path in the config.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    class_names = list(config["classes"])
    rules = load_rules(config["rules"])
    missing_rules = [name for name in class_names if name not in rules["name_to_stream"]]
    if missing_rules:
        raise ValueError(f"config classes missing from rules.yaml: {missing_rules}")

    item_log = resolve(args.item_log or config["item_log"]).resolve()
    image_root = resolve(config["image_root"])
    log_rows = read_item_log(item_log)
    captures = scan_captures(image_root)
    rows = build_rows(log_rows, captures, set(class_names))
    write_manifest(config["manifest"], rows)

    frame_counts: dict[str, int] = defaultdict(int)
    item_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        frame_counts[row["label"]] += 1
    for row in log_rows:
        item_counts[row["label"]] += 1

    print(f"wrote {len(rows)} frames for {len(log_rows)} items to {resolve(config['manifest'])}")
    print(f"item log: {item_log}")
    print("material class -> stream")
    for name in class_names:
        stream = rules["name_to_stream"][name]
        print(
            f"  {name:22} {stream:10} {item_counts[name]:4d} items {frame_counts[name]:4d} frames"
        )


if __name__ == "__main__":
    main()
