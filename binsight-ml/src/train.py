"""Two-phase fine-tune of one manifest.

Phase 1 trains the head with the backbone frozen.
Phase 2 unfreezes the top fraction of backbone layers at a much smaller
learning rate and stops early on validation loss.
The backbone call is training=False in both phases (see model.py).

Usage (from binsight-ml/):
    python src/train.py --config configs/trashnet.yaml --smoke
    python src/train.py --config configs/trashnet.yaml

The full run is meant for a GPU (Colab T4). --smoke uses a few images per
class and one epoch per phase so a CPU can prove the pipeline connects.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from src.data import batches_of, class_weights, make_dataset
from src.evaluate import evaluate_predictions, predict_labels, write_metrics
from src.model import build_model, compile_model, set_backbone_trainable, trainable_parameter_count
from src.util import (
    count_by_label,
    group_items,
    limit_items_per_class,
    load_rules,
    load_yaml,
    resolve,
    split_item_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from a manifest config.")
    parser.add_argument("--config", default="configs/trashnet.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="8 images per class, 1 epoch per phase, to check the pipeline on CPU.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    train_cfg = dict(config["train"])
    batch_size = int(config["batch_size"])
    per_class = None
    if args.smoke:
        train_cfg["phase1_epochs"] = 1
        train_cfg["phase2_epochs"] = 1
        batch_size = min(batch_size, 8)
        per_class = 8

    class_names = list(config["classes"])
    rules = load_rules(config["rules"])
    _check_classes(class_names, rules)

    rows = _read_rows(config["manifest"])
    unknown = sorted({row["label"] for row in rows} - set(class_names))
    if unknown:
        raise ValueError(f"manifest labels missing from config classes: {unknown}")

    items = group_items(rows)
    if per_class is not None:
        items = limit_items_per_class(items, per_class, int(config["seed"]))
    split_ids = split_item_ids(
        items,
        float(config["val_fraction"]),
        float(config["test_fraction"]),
        int(config["seed"]),
    )
    splits = batches_of(split_ids, items)

    run_name = args.run_name or (f"{config['name']}_smoke" if args.smoke else config["name"])
    run_dir = resolve("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_splits(run_dir / "splits.csv", splits)
    (run_dir / "class_names.json").write_text(json.dumps(class_names, indent=2) + "\n")

    tf.keras.utils.set_random_seed(int(config["seed"]))
    class_to_index = {name: i for i, name in enumerate(class_names)}
    image_root = resolve(config["image_root"])
    train_ds = make_dataset(
        splits["train"], image_root, class_to_index, int(config["image_size"]),
        batch_size, True, config["augment"], int(config["seed"]),
    )
    val_ds = make_dataset(
        splits["val"], image_root, class_to_index, int(config["image_size"]),
        batch_size, False, config["augment"], int(config["seed"]),
    )
    test_ds = make_dataset(
        splits["test"], image_root, class_to_index, int(config["image_size"]),
        batch_size, False, config["augment"], int(config["seed"]),
    )
    _check_batch(train_ds, int(config["image_size"]))
    _check_split_modes(
        splits, image_root, class_to_index, int(config["image_size"]), config["augment"], int(config["seed"])
    )

    weights = class_weights(splits["train"], class_names)
    model_cfg = config["model"]
    model, backbone = build_model(
        num_classes=len(class_names),
        image_size=int(config["image_size"]),
        alpha=float(model_cfg["alpha"]),
        dropout=float(model_cfg["dropout"]),
        weights=model_cfg["weights"],
    )

    frozen = set_backbone_trainable(backbone, 0.0)
    compile_model(model, float(train_cfg["phase1_lr"]))
    print(
        f"phase 1: backbone layers trainable={frozen}, "
        f"parameters={trainable_parameter_count(model)}"
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(train_cfg["phase1_epochs"]),
        class_weight=weights,
        shuffle=False,
        verbose=2,
    )
    phase1_path = run_dir / "phase1.keras"
    model.save(phase1_path)
    phase1_val = float(history1.history["val_loss"][-1])

    trainable_layers = set_backbone_trainable(backbone, float(train_cfg["unfreeze_fraction"]))
    compile_model(model, float(train_cfg["phase2_lr"]))
    print(
        f"phase 2: backbone layers trainable={trainable_layers}, "
        f"parameters={trainable_parameter_count(model)}"
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(train_cfg["phase2_epochs"]),
        class_weight=weights,
        shuffle=False,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(train_cfg["early_stopping_patience"]),
                restore_best_weights=True,
            )
        ],
    )
    phase2_val = float(min(history2.history["val_loss"]))
    kept = "phase2"
    if phase2_val > phase1_val:
        model = tf.keras.models.load_model(phase1_path)
        kept = "phase1"
    phase1_path.unlink()
    model.save(run_dir / "model.keras")

    y_true, y_pred = predict_labels(model, test_ds)
    scored = evaluate_predictions(y_true, y_pred, class_names, rules)
    per_class_counts = {
        name: {split: count_by_label(splits[split]).get(name, 0) for split in ("train", "val", "test")}
        for name in class_names
    }
    metrics = {
        "run_name": run_name,
        "smoke": bool(args.smoke),
        "config": args.config,
        "rules_version": rules["version"],
        "weights_kept": kept,
        "classes": class_names,
        "split": {
            "strategy": "grouped by item_id, stratified by label",
            "seed": int(config["seed"]),
            "fractions": {
                "val": float(config["val_fraction"]),
                "test": float(config["test_fraction"]),
            },
            "counts": {name: len(splits[name]) for name in ("train", "val", "test")},
            "per_class_counts": per_class_counts,
        },
        "class_weights": {class_names[index]: weight for index, weight in weights.items()},
        "training": {
            "phase1_epochs_ran": len(history1.history["loss"]),
            "phase1_val_loss": phase1_val,
            "phase2_epochs_ran": len(history2.history["loss"]),
            "phase2_best_val_loss": phase2_val,
            "phase2_history": {
                key: [float(value) for value in values]
                for key, values in history2.history.items()
            },
        },
        "evaluation_notes": list(config.get("evaluation_notes", [])),
        **scored,
    }
    write_metrics(run_dir, metrics)
    print(f"kept {kept} weights")
    print(f"material macro-F1 {scored['material']['macro_f1']:.4f}")
    print(f"stream macro-F1   {scored['stream']['macro_f1']:.4f}")
    print(f"zero-support streams: {scored['stream']['zero_support'] or 'none'}")
    print(f"wrote {run_dir / 'metrics.json'}")


def _check_classes(class_names: list[str], rules: dict) -> None:
    if len(class_names) != len(set(class_names)):
        raise ValueError("config classes contains a duplicate")
    missing = [name for name in class_names if name not in rules["name_to_stream"]]
    if missing:
        raise ValueError(f"config classes missing from rules.yaml: {missing}")


def _read_rows(path: str) -> list[dict]:
    from src.util import read_manifest

    return read_manifest(path)


def _write_splits(path: Path, splits: dict[str, list[dict]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "filepath", "item_id", "label"),
            lineterminator="\n",
        )
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for row in splits[split_name]:
                writer.writerow({"split": split_name, **row})


def _check_split_modes(splits, image_root, class_to_index, image_size, augment, seed) -> None:
    """Val is a fixed center-crop. Training augmentation must actually change pixels."""
    common = dict(
        image_root=image_root,
        class_to_index=class_to_index,
        image_size=image_size,
        batch_size=1,
        augment=augment,
        seed=seed,
    )
    val_ds = make_dataset(splits["val"][:1], training=False, **common)
    val_a = next(iter(val_ds))[0].numpy()
    val_b = next(iter(val_ds))[0].numpy()
    if not np.array_equal(val_a, val_b):
        raise RuntimeError("validation preprocessing changed between passes")
    train_ds = make_dataset(splits["train"][:1], training=True, **common)
    train_a = next(iter(train_ds))[0].numpy()
    train_b = next(iter(train_ds))[0].numpy()
    if np.array_equal(train_a, train_b):
        raise RuntimeError("training augmentation produced identical images on two passes")
    print("preprocessing ok: val is deterministic, train augmentation changes pixels")


def _check_batch(dataset, image_size: int) -> None:
    images, labels = next(iter(dataset))
    if tuple(images.shape[1:]) != (image_size, image_size, 3):
        raise RuntimeError(f"expected images (N, {image_size}, {image_size}, 3), got {images.shape}")
    low = float(tf.reduce_min(images))
    high = float(tf.reduce_max(images))
    if low < -1.01 or high > 1.01:
        raise RuntimeError(f"preprocess_input should land in [-1, 1], got [{low}, {high}]")
    print(f"batch ok: shape={tuple(images.shape)} range=[{low:.3f}, {high:.3f}] labels={labels.numpy().tolist()}")


if __name__ == "__main__":
    main()
