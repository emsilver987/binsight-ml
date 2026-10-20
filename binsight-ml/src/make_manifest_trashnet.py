"""Build the TrashNet manifest.

TrashNet ships as class folders and has no item ids. This script is the only
TrashNet-specific stage: it writes filepath, item_id, label rows whose labels
are rules.yaml class names. Later stages never inspect the folder names.

item_id is the path of that one image. The shared item-level splitter then
reduces to a stratified random image split. TrashNet cannot do better than
that, and the random split can flatter the baseline.

Usage (from binsight-ml/):
    python src/make_manifest_trashnet.py --config configs/trashnet.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.util import IMAGE_SUFFIXES, load_rules, load_yaml, resolve, write_manifest


def build_rows(dataset_dir: Path, label_map: dict[str, str]) -> list[dict]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"TrashNet directory not found: {dataset_dir}\n"
            "Download dataset-resized.zip from\n"
            "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip\n"
            "and unzip it so the six class folders sit in that directory."
        )

    folders = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    unknown = [name for name in folders if name not in label_map]
    missing = [name for name in label_map if name not in folders]
    if unknown or missing:
        raise ValueError(
            "TrashNet folders do not match trashnet_label_map. "
            f"unknown folders: {unknown or 'none'}; "
            f"missing folders: {missing or 'none'}"
        )

    rows = []
    for folder, material in label_map.items():
        for path in sorted((dataset_dir / folder).iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(dataset_dir).as_posix()
            rows.append(
                {
                    "filepath": relative,
                    "item_id": path.relative_to(dataset_dir).with_suffix("").as_posix(),
                    "label": material,
                }
            )
    if not rows:
        raise ValueError(f"no images found under {dataset_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the TrashNet manifest.")
    parser.add_argument("--config", default="configs/trashnet.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    label_map = config["trashnet_label_map"]
    dataset_dir = resolve(config["image_root"])
    rules = load_rules(config["rules"])

    unknown_labels = sorted({label for label in label_map.values() if label not in rules["name_to_stream"]})
    if unknown_labels:
        raise ValueError(f"label map points at names that are not in rules.yaml: {unknown_labels}")

    rows = build_rows(dataset_dir, label_map)
    write_manifest(config["manifest"], rows)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["label"]] += 1

    print(f"wrote {len(rows)} rows to {resolve(config['manifest'])}")
    print("TrashNet folder -> material class -> stream")
    for folder in sorted(label_map):
        material = label_map[folder]
        stream = rules["name_to_stream"][material]
        print(f"  {folder:12} {material:22} {stream:10} {counts[material]:4d} images")

    reachable = set(label_map.values())
    silent = [name for name in rules["classes"] if name not in reachable]
    print("material classes with no TrashNet images:")
    for name in silent:
        print(f"  {name:22} {rules['name_to_stream'][name]}")
    print("item_id is the image path. TrashNet has no physical-item ids.")


if __name__ == "__main__":
    main()
