"""Material-level and stream-level scores for one run.

Material macro-F1 is over the classes this model was trained to emit.
Stream macro-F1 maps those predictions through rules.yaml and averages
Recycle, Trash, Compost, and Special. A stream with no test rows scores 0,
which is what makes a 6-class TrashNet model comparable to the 10-class model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import make_dataset
from src.util import load_rules, load_yaml, resolve


def scores(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    per_label = {}
    f1_values = []
    for index, label in enumerate(labels):
        true_mask = y_true == index
        pred_mask = y_pred == index
        tp = int(np.count_nonzero(true_mask & pred_mask))
        fp = int(np.count_nonzero(~true_mask & pred_mask))
        fn = int(np.count_nonzero(true_mask & ~pred_mask))
        support = int(np.count_nonzero(true_mask))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
    macro = float(np.mean(f1_values)) if f1_values else 0.0
    return {"macro_f1": macro, "per_label": per_label}


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> list[list[int]]:
    matrix = np.zeros((n_labels, n_labels), dtype=np.int64)
    for true_index, pred_index in zip(y_true, y_pred):
        matrix[int(true_index), int(pred_index)] += 1
    return matrix.tolist()


def predict_labels(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    true_parts = []
    pred_parts = []
    for images, labels in dataset:
        probabilities = model(images, training=False).numpy()
        true_parts.append(labels.numpy())
        pred_parts.append(np.argmax(probabilities, axis=1))
    if not true_parts:
        raise ValueError("evaluation dataset produced no batches")
    return np.concatenate(true_parts), np.concatenate(pred_parts)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    rules: dict,
) -> dict:
    material = scores(y_true, y_pred, class_names)
    material_matrix = confusion(y_true, y_pred, len(class_names))

    name_to_stream = rules["name_to_stream"]
    stream_names = rules["streams"]
    stream_index = {name: i for i, name in enumerate(stream_names)}
    true_streams = np.array([stream_index[name_to_stream[class_names[i]]] for i in y_true])
    pred_streams = np.array([stream_index[name_to_stream[class_names[i]]] for i in y_pred])
    stream = scores(true_streams, pred_streams, stream_names)
    stream_matrix = confusion(true_streams, pred_streams, len(stream_names))
    zero_support = [
        name for name, stats in stream["per_label"].items() if stats["support"] == 0
    ]
    return {
        "material": {
            "macro_f1": material["macro_f1"],
            "labels": class_names,
            "per_class": material["per_label"],
            "confusion_matrix": material_matrix,
        },
        "stream": {
            "macro_f1": stream["macro_f1"],
            "labels": stream_names,
            "per_stream": stream["per_label"],
            "confusion_matrix": stream_matrix,
            "zero_support": zero_support,
            "zero_division": 0,
        },
        "material_to_stream": {name: name_to_stream[name] for name in class_names},
    }


def write_metrics(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_confusion_csv(
        run_dir / "confusion_matrix.csv",
        payload["material"]["confusion_matrix"],
        payload["material"]["labels"],
    )
    _write_confusion_png(
        run_dir / "confusion_matrix.png",
        payload["material"]["confusion_matrix"],
        payload["material"]["labels"],
        "Material confusion matrix",
    )
    _write_confusion_png(
        run_dir / "stream_confusion.png",
        payload["stream"]["confusion_matrix"],
        payload["stream"]["labels"],
        "Stream confusion matrix",
    )


def _write_confusion_csv(path: Path, matrix: list[list[int]], labels: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def _write_confusion_png(path: Path, matrix: list[list[int]], labels: list[str], title: str) -> None:
    array = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(array, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(array[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score a saved run on its test split.")
    parser.add_argument("--run", required=True, help="Run directory, e.g. runs/trashnet")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_dir = resolve(args.run)
    config = load_yaml(args.config)
    class_names = json.loads((run_dir / "class_names.json").read_text())
    rules = load_rules(config["rules"])
    rows = _read_split_csv(run_dir / "splits.csv", "test")
    dataset = make_dataset(
        rows,
        resolve(config["image_root"]),
        {name: i for i, name in enumerate(class_names)},
        int(config["image_size"]),
        int(config["batch_size"]),
        training=False,
        augment=config["augment"],
        seed=int(config["seed"]),
    )
    model = tf.keras.models.load_model(run_dir / "model.keras")
    y_true, y_pred = predict_labels(model, dataset)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    scored = evaluate_predictions(y_true, y_pred, class_names, rules)
    metrics.update(scored)
    write_metrics(run_dir, metrics)
    print(f"material macro-F1 {scored['material']['macro_f1']:.4f}")
    print(f"stream macro-F1   {scored['stream']['macro_f1']:.4f}")
    print(f"wrote {run_dir / 'metrics.json'}")


def _read_split_csv(path: Path, split_name: str) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split_name:
                rows.append(
                    {
                        "filepath": row["filepath"],
                        "item_id": row["item_id"],
                        "label": row["label"],
                    }
                )
    if not rows:
        raise ValueError(f"no {split_name} rows in {path}")
    return rows


if __name__ == "__main__":
    main()
