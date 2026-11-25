"""Choose stream confidence thresholds on the validation split.

Deployment matches rules.yaml. Average the last vote_window frame
probabilities, sum the materials that share a stream, and take that
stream's probability as the confidence of the argmax stream. Accept the
stream when its confidence clears its threshold. Otherwise the decision
is Unknown and the lids stay closed.

The test split is not read. Selection is item-level: one capture, one vote.

Recycle is the strict stream. A wrong Recycle on the validation set is
never accepted. Correct Recycle votes are kept only when they sit with
the confident cluster: walking up from the weakest, a vote more than
0.15 below the next correct vote does not get to pull the threshold
down, and that item abstains. Trash and Compost take the highest
threshold that still accepts every correct vote, and they cannot sit
above Recycle. Special takes the highest threshold that still accepts
every true Special vote, including a weak one, because a missed battery
is the failure that comment in rules.yaml is about. A false Special
only closes the lids, so it does not raise that bar. After those
choices, a true Special item that would still open a lid pushes that
lid's threshold up until the item becomes Unknown.

Usage (from binsight-ml/):
    python src/tune_thresholds.py --config configs/binsight.yaml --run runs/binsight
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from src.data import make_dataset
from src.util import load_rules, load_yaml, resolve

LID_STREAMS = ("Recycle", "Trash", "Compost")
# A correct Recycle vote this far below the next one is too weak to hold the bar down.
RECYCLE_OUTLIER_GAP = 0.15
THRESHOLD_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<name>Recycle|Trash|Compost|Special):"
    r"(?P<gap>\s*)(?P<value>null|[0-9]*\.?[0-9]+)(?P<rest>\s*(#.*)?)$"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set stream thresholds from the validation split.")
    parser.add_argument("--config", default="configs/binsight.yaml")
    parser.add_argument("--run", default="runs/binsight")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the chosen thresholds without editing rules.yaml.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_dir = resolve(args.run)
    rules_path = resolve(config["rules"])
    raw_rules = load_yaml(rules_path)
    rules = load_rules(rules_path)
    vote_window = int(raw_rules["decision"]["vote_window"])
    stream_names = list(rules["streams"])
    class_names = json.loads((run_dir / "class_names.json").read_text())
    rows = _read_split(run_dir / "splits.csv", "val")

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
    probabilities = _predict(model, dataset, rows, class_names)
    items = vote_items(
        rows,
        probabilities,
        class_names,
        rules["name_to_stream"],
        stream_names,
        vote_window,
    )
    chosen, trace = select_thresholds(items, stream_names)
    decisions = apply_thresholds(items, chosen)

    payload = {
        "split": "val",
        "vote_window": vote_window,
        "n_items": len(items),
        "n_frames": len(rows),
        "thresholds": chosen,
        "trace": trace,
        "items": decisions,
    }
    out_path = run_dir / "thresholds.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    _print_report(items, decisions, chosen, trace, out_path)

    if args.dry_run:
        print(f"dry run: left {rules_path} unchanged")
        return
    _write_rules(rules_path, chosen)
    print(f"wrote thresholds to {rules_path}")


def vote_items(
    rows: list[dict],
    probabilities: np.ndarray,
    class_names: list[str],
    name_to_stream: dict[str, str],
    stream_names: list[str],
    vote_window: int,
) -> list[dict]:
    """One decision per item: mean of the last vote_window frames, then sum to streams."""
    if vote_window < 1:
        raise ValueError("vote_window must be at least 1")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["item_id"]].append(index)

    stream_of_class = [name_to_stream[name] for name in class_names]
    items = []
    for item_id in sorted(grouped):
        indexes = sorted(grouped[item_id], key=lambda index: rows[index]["filepath"])
        window = indexes[-vote_window:]
        labels = {rows[index]["label"] for index in window}
        if len(labels) != 1:
            raise ValueError(f"item {item_id} mixes labels {sorted(labels)}")
        label = next(iter(labels))
        mean = probabilities[window].mean(axis=0)
        stream_probs = {name: 0.0 for name in stream_names}
        for class_index, mass in enumerate(mean):
            stream_probs[stream_of_class[class_index]] += float(mass)
        total = sum(stream_probs.values())
        if abs(total - 1.0) > 1e-4:
            raise RuntimeError(f"item {item_id} stream probabilities sum to {total}")
        pred_stream = max(stream_names, key=lambda name: stream_probs[name])
        items.append(
            {
                "item_id": item_id,
                "label": label,
                "true_stream": name_to_stream[label],
                "pred_stream": pred_stream,
                "confidence": stream_probs[pred_stream],
                "stream_probs": stream_probs,
                "n_frames": len(window),
            }
        )
    return items


def select_thresholds(items: list[dict], stream_names: list[str]) -> tuple[dict[str, float], list[str]]:
    missing = [name for name in ("Recycle", "Trash", "Compost", "Special") if name not in stream_names]
    if missing:
        raise ValueError(f"rules.yaml is missing streams: {missing}")

    grid = [round(step / 100, 2) for step in range(50, 101)]
    trace = []
    recycle, recycle_note = _choose_recycle(items, grid)
    trace.append(recycle_note)
    thresholds = {"Recycle": recycle}
    for stream in ("Trash", "Compost"):
        value, note = _choose_cover(items, stream, grid, ceiling=recycle)
        thresholds[stream] = value
        trace.append(note)
    special, special_note = _choose_cover(items, "Special", grid, ceiling=1.0)
    thresholds["Special"] = special
    trace.append(special_note)
    thresholds, guard_notes = _block_special_from_opening(items, thresholds)
    trace.extend(guard_notes)
    return thresholds, trace


def apply_thresholds(items: list[dict], thresholds: dict[str, float]) -> list[dict]:
    decided = []
    for item in items:
        accepted = item["confidence"] + 1e-9 >= thresholds[item["pred_stream"]]
        decision = item["pred_stream"] if accepted else "Unknown"
        decided.append(
            {
                "item_id": item["item_id"],
                "label": item["label"],
                "true_stream": item["true_stream"],
                "pred_stream": item["pred_stream"],
                "confidence": round(item["confidence"], 4),
                "decision": decision,
                "correct": decision == item["true_stream"],
                "stream_probs": {name: round(prob, 4) for name, prob in item["stream_probs"].items()},
                "n_frames": item["n_frames"],
            }
        )
    return decided


def _choose_recycle(items: list[dict], grid: list[float]) -> tuple[float, str]:
    """Highest bar that keeps the confident correct Recycle cluster and rejects every wrong one."""
    correct = _confidences(items, "Recycle", correct=True)
    wrong = _confidences(items, "Recycle", correct=False)
    kept, held_out = _drop_recycle_outliers(correct)
    if not kept:
        threshold = _step_above(max(wrong)) if wrong else 0.99
        return min(1.0, threshold), f"Recycle: {threshold:.2f} has no confident correct validation vote"
    threshold = _highest_cover(grid, floor=min(kept), false_max=max(wrong) if wrong else None, ceiling=1.0)
    if threshold is None:
        threshold = min(1.0, _step_above(max(wrong)))
        return threshold, (
            f"Recycle: {threshold:.2f} sits above a wrong vote that outranks the correct cluster"
        )
    held = ""
    if held_out:
        listed = ", ".join(f"{confidence:.3f}" for confidence in held_out)
        held = f"; left below the bar: {listed}"
    return threshold, (
        f"Recycle: {threshold:.2f} keeps {len(kept)} confident votes, "
        f"rejects {len(wrong)} wrong{held}"
    )


def _choose_cover(
    items: list[dict], stream: str, grid: list[float], ceiling: float
) -> tuple[float, str]:
    """Highest threshold that still accepts every correct vote for this stream."""
    correct = _confidences(items, stream, correct=True)
    wrong = _confidences(items, stream, correct=False)
    if not correct:
        threshold = min(0.50, ceiling)
        return threshold, f"{stream}: {threshold:.2f}, no validation item was predicted {stream}"
    threshold = _highest_cover(grid, floor=min(correct), false_max=max(wrong) if wrong else None, ceiling=ceiling)
    if threshold is None:
        threshold = min(ceiling, _step_above(max(wrong)))
        return threshold, (
            f"{stream}: {threshold:.2f} rejects a wrong vote that outranks every correct one"
        )
    return threshold, (
        f"{stream}: {threshold:.2f} keeps all {len(correct)} correct votes, "
        f"rejects {len(wrong)} wrong, ceiling {ceiling:.2f}"
    )


def _drop_recycle_outliers(confidences: list[float]) -> tuple[list[float], list[float]]:
    """Drop a weak prefix whose gap to the next correct vote exceeds RECYCLE_OUTLIER_GAP."""
    ordered = sorted(confidences)
    start = 0
    while start + 1 < len(ordered) and ordered[start + 1] - ordered[start] > RECYCLE_OUTLIER_GAP:
        start += 1
    return ordered[start:], ordered[:start]


def _highest_cover(
    grid: list[float], floor: float, false_max: float | None, ceiling: float
) -> float | None:
    """Highest grid point that still accepts `floor` and rejects `false_max`."""
    limit = min(ceiling, floor)
    candidates = [
        threshold
        for threshold in grid
        if threshold <= limit + 1e-9 and (false_max is None or threshold > false_max + 1e-9)
    ]
    if not candidates:
        return None
    return max(candidates)


def _confidences(items: list[dict], stream: str, correct: bool) -> list[float]:
    return [
        item["confidence"]
        for item in items
        if item["pred_stream"] == stream and (item["true_stream"] == stream) is correct
    ]


def _block_special_from_opening(items: list[dict], thresholds: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """A battery or circuit must not open a lid on the validation set."""
    updated = dict(thresholds)
    notes = []
    for item in items:
        if item["true_stream"] != "Special":
            continue
        stream = item["pred_stream"]
        if stream not in LID_STREAMS:
            continue
        if item["confidence"] + 1e-9 < updated[stream]:
            continue
        raised = min(1.0, _step_above(item["confidence"]))
        if raised <= updated[stream]:
            continue
        notes.append(
            f"raised {stream} from {updated[stream]:.2f} to {raised:.2f} "
            f"so item {item['item_id']} (Special) stays closed"
        )
        updated[stream] = raised
    for stream in ("Trash", "Compost"):
        if updated[stream] > updated["Recycle"] + 1e-9:
            notes.append(
                f"raised Recycle from {updated['Recycle']:.2f} to {updated[stream]:.2f} "
                f"so it stays at least as strict as {stream}"
            )
            updated["Recycle"] = updated[stream]
    if not notes:
        notes.append("no validation Special item would have opened a lid")
    return updated, notes


def _step_above(confidence: float) -> float:
    stepped = math.floor((confidence + 1e-8) * 100 + 1) / 100
    return round(min(1.0, stepped), 2)


def _predict(model, dataset, rows: list[dict], class_names: list[str]) -> np.ndarray:
    batches = []
    labels = []
    for images, batch_labels in dataset:
        batches.append(model(images, training=False).numpy())
        labels.append(batch_labels.numpy())
    if not batches:
        raise ValueError("validation split produced no frames")
    probabilities = np.concatenate(batches)
    seen = np.concatenate(labels)
    expected = np.array([class_names.index(row["label"]) for row in rows])
    if len(seen) != len(rows) or not np.array_equal(seen, expected):
        raise RuntimeError("prediction order does not match the validation rows")
    return probabilities


def _read_split(path: Path, split_name: str) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split_name:
                continue
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


def _print_report(items, decisions, thresholds, trace, out_path: Path) -> None:
    print(f"{'item':<6} {'true':<10} {'pred':<10} {'conf':>6}  decision")
    for item in decisions:
        mark = "ok" if item["correct"] else "  "
        print(
            f"{item['item_id']:<6} {item['true_stream']:<10} {item['pred_stream']:<10} "
            f"{item['confidence']:6.3f}  {item['decision']:<8} {mark}  {item['label']}"
        )
    unknown = [item for item in decisions if item["decision"] == "Unknown"]
    opened = [item for item in decisions if item["decision"] in LID_STREAMS]
    wrong_open = [item for item in opened if item["decision"] != item["true_stream"]]
    print(
        f"items {len(items)}  decided {len(items) - len(unknown)}  "
        f"unknown {len(unknown)}  lids open {len(opened)}  wrong lid {len(wrong_open)}"
    )
    for line in trace:
        print(line)
    print(
        "thresholds  "
        + "  ".join(f"{name} {thresholds[name]:.2f}" for name in ("Recycle", "Trash", "Compost", "Special"))
    )
    print(f"wrote {out_path}")


def _write_rules(path: Path, thresholds: dict[str, float]) -> None:
    lines = path.read_text().splitlines(keepends=True)
    replaced = set()
    for index, line in enumerate(lines):
        match = THRESHOLD_LINE.match(line.rstrip("\n"))
        if match is None or match.group("name") not in thresholds:
            continue
        if match.group("name") in replaced:
            raise RuntimeError(f"more than one threshold line for {match.group('name')} in {path}")
        newline = "\n" if line.endswith("\n") else ""
        rest = match.group("rest") or ""
        lines[index] = (
            f"{match.group('indent')}{match.group('name')}:{match.group('gap')}"
            f"{thresholds[match.group('name')]:.2f}{rest}{newline}"
        )
        replaced.add(match.group("name"))
    missing = [name for name in thresholds if name not in replaced]
    if missing:
        raise RuntimeError(f"{path} has no threshold line for {missing}")
    path.write_text("".join(lines))


if __name__ == "__main__":
    main()
