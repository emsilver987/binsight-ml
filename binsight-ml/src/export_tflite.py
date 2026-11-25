"""Export the trained Keras model as an int8 TFLite model for the XIAO.

The validation path in src/data.py is the reference: center-crop, resize to
96x96, then MobileNetV2 preprocessing. Post-training quantization uses the
training split only. The script also writes the Arduino headers the classify
sketch includes, and checks the int8 model against the validation items.

Usage (from binsight-ml/):
    python src/export_tflite.py --config configs/binsight.yaml --run runs/binsight
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from src.util import IMAGE_SUFFIXES, load_rules, load_yaml, resolve

SKETCH_DIR = PROJECT / "arduino" / "XAIO_ESP32S3_classify"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the BinSight model for the XIAO.")
    parser.add_argument("--config", default="configs/binsight.yaml")
    parser.add_argument("--run", default="runs/binsight")
    parser.add_argument("--max-representative", type=int, default=200)
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_dir = resolve(args.run)
    rules_path = resolve(config["rules"])
    raw_rules = load_yaml(rules_path)
    rules = load_rules(rules_path)
    class_names = json.loads((run_dir / "class_names.json").read_text())
    image_size = int(config["image_size"])
    image_root = resolve(config["image_root"])
    train_rows = _read_split(run_dir / "splits.csv", "train")
    val_rows = _read_split(run_dir / "splits.csv", "val")
    representative_rows = _stratified_cap(train_rows, int(args.max_representative), int(config["seed"]))

    print(f"loading {run_dir / 'model.keras'}")
    model = tf.keras.models.load_model(run_dir / "model.keras")
    samples = [
        _preprocess(image_root / row["filepath"], image_size) for row in representative_rows
    ]

    def representative():
        for sample in samples:
            yield [sample[None, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    print(f"quantizing on {len(samples)} training images")
    tflite_bytes = converter.convert()

    tflite_path = run_dir / "model_int8.tflite"
    tflite_path.write_bytes(tflite_bytes)
    print(f"wrote {tflite_path} ({len(tflite_bytes) / 1024:.1f} KB)")

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    ops = _op_names(tflite_bytes)
    print("ops:", ", ".join(ops))
    print(
        f"input {input_detail['dtype'].__name__} scale={input_detail['quantization'][0]} "
        f"zero={input_detail['quantization'][1]}"
    )
    print(
        f"output {output_detail['dtype'].__name__} scale={output_detail['quantization'][0]} "
        f"zero={output_detail['quantization'][1]}"
    )
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise RuntimeError("conversion did not produce int8 input and output")
    if tuple(input_detail["shape"]) != (1, image_size, image_size, 3):
        raise RuntimeError(f"unexpected input shape {input_detail['shape']}")
    if int(output_detail["shape"][-1]) != len(class_names):
        raise RuntimeError("output width does not match class_names.json")

    expected = {
        item["item_id"]: item["decision"]
        for item in json.loads((run_dir / "thresholds.json").read_text())["items"]
    }
    _check_validation(
        interpreter,
        input_detail,
        output_detail,
        image_root,
        image_size,
        val_rows,
        class_names,
        rules,
        raw_rules,
        expected,
    )

    SKETCH_DIR.mkdir(parents=True, exist_ok=True)
    _write_model_header(SKETCH_DIR / "binsight_model.h", tflite_bytes, ops)
    _write_labels_header(SKETCH_DIR / "binsight_labels.h", class_names, rules, raw_rules, image_size)
    print(f"wrote {SKETCH_DIR / 'binsight_model.h'}")
    print(f"wrote {SKETCH_DIR / 'binsight_labels.h'}")


def _preprocess(path: Path, image_size: int) -> np.ndarray:
    """Match the validation loader in src/data.py."""
    image = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)
    side = tf.minimum(shape[0], shape[1])
    offset_y = (shape[0] - side) // 2
    offset_x = (shape[1] - side) // 2
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, side, side)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image.numpy().astype(np.float32)


def _quantize(image: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero = detail["quantization"]
    quantized = np.round(image / scale + zero)
    return np.clip(quantized, -128, 127).astype(np.int8)


def _dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero = detail["quantization"]
    return (values.astype(np.float32) - zero) * scale


def _predict_item(interpreter, input_detail, output_detail, images: list[np.ndarray]) -> np.ndarray:
    accumulated = np.zeros(int(output_detail["shape"][-1]), dtype=np.float64)
    for image in images:
        interpreter.set_tensor(input_detail["index"], _quantize(image, input_detail)[None, ...])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0]
        accumulated += _dequantize(raw, output_detail)
    return accumulated / len(images)


def _check_validation(
    interpreter,
    input_detail,
    output_detail,
    image_root: Path,
    image_size: int,
    rows: list[dict],
    class_names: list[str],
    rules: dict,
    raw_rules: dict,
    expected: dict[str, str],
) -> None:
    vote_window = int(raw_rules["decision"]["vote_window"])
    thresholds = raw_rules["decision"]["thresholds"]
    stream_names = list(rules["streams"])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["item_id"]].append(row)

    matched = 0
    print(f"{'item':<6} {'int8':<10} {'keras':<10} {'conf':>6}  code")
    for item_id in sorted(grouped):
        item_rows = sorted(grouped[item_id], key=lambda row: row["filepath"])[-vote_window:]
        images = [_preprocess(image_root / row["filepath"], image_size) for row in item_rows]
        mean = _predict_item(interpreter, input_detail, output_detail, images)
        stream_probs = {name: 0.0 for name in stream_names}
        for index, mass in enumerate(mean):
            stream_probs[rules["name_to_stream"][class_names[index]]] += float(mass)
        pred_stream = max(stream_names, key=lambda name: stream_probs[name])
        confidence = stream_probs[pred_stream]
        accepted = confidence + 1e-9 >= float(thresholds[pred_stream])
        decision = pred_stream if accepted else "Unknown"
        code = raw_rules["streams"][decision]["code"]
        keras = expected.get(item_id, "?")
        mark = "ok" if decision == keras else "DIFF"
        if decision == keras:
            matched += 1
        print(f"{item_id:<6} {decision:<10} {keras:<10} {confidence:6.3f}  {code}  {mark}")
    print(f"int8 agrees with the Keras validation decisions on {matched}/{len(grouped)} items")


def _op_names(tflite_bytes: bytes) -> list[str]:
    details = tf.lite.Interpreter(model_content=tflite_bytes)._get_ops_details()
    names = []
    for op in details:
        name = op["op_name"]
        if name not in names:
            names.append(name)
    return names


def _add_method(op_name: str) -> str:
    parts = []
    for part in op_name.split("_"):
        if len(part) > 1 and part[0].isalpha():
            parts.append(part[0].upper() + part[1:].lower())
        else:
            parts.append(part.upper())
    formatted = "".join(parts).replace("Lstm", "LSTM").replace("BatchMatmul", "BatchMatMul")
    return "Add" + formatted


def _write_model_header(path: Path, tflite_bytes: bytes, ops: list[str]) -> None:
    methods = [_add_method(name) for name in ops]
    hex_lines = []
    for offset in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[offset : offset + 12]
        hex_lines.append(", ".join(f"0x{byte:02x}" for byte in chunk))
    body = ",\n  ".join(hex_lines)
    calls = "\n".join(f"  micro_op_resolver.{method}();" for method in methods)
    path.write_text(
        f"""// Generated by binsight-ml/src/export_tflite.py. Do not edit.
// Model bytes are int8 MobileNetV2 for the XIAO classify sketch.
// Setup uses PSRAM for the arena; see XAIO_ESP32S3_classify.ino.

#pragma once

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

constexpr int TFLMnumberOperators = {len(methods)};

inline tflite::MicroMutableOpResolver<TFLMnumberOperators> TFLMgetResolver() {{
  tflite::MicroMutableOpResolver<TFLMnumberOperators> micro_op_resolver;
{calls}
  return micro_op_resolver;
}}

#if defined(__GNUC__)
#define BINSIGHT_MODEL_ALIGN __attribute__((aligned(16)))
#else
#define BINSIGHT_MODEL_ALIGN
#endif

const unsigned int TFLMmodelLength = {len(tflite_bytes)};

const unsigned char TFLM_binsight_model[] BINSIGHT_MODEL_ALIGN = {{
  {body}
}};
"""
    )


def _write_labels_header(
    path: Path,
    class_names: list[str],
    rules: dict,
    raw_rules: dict,
    image_size: int,
) -> None:
    stream_names = list(rules["streams"])
    stream_index = {name: index for index, name in enumerate(stream_names)}
    class_to_stream = [stream_index[rules["name_to_stream"][name]] for name in class_names]
    thresholds = raw_rules["decision"]["thresholds"]
    missing = [name for name in stream_names if thresholds.get(name) is None]
    if missing:
        raise ValueError(f"rules.yaml thresholds are unset: {missing}")
    vote_window = int(raw_rules["decision"]["vote_window"])

    def c_string(value: str) -> str:
        return json.dumps(value)

    codes = []
    for name in stream_names:
        code = raw_rules["streams"][name]["code"]
        if len(code) != 1:
            raise ValueError(f"stream {name} code {code!r} is not one character")
        codes.append(f"'{code}'")

    path.write_text(
        f"""// Generated by binsight-ml/src/export_tflite.py from rules.yaml and class_names.json.
#pragma once

#include <stdint.h>

constexpr int kImageSize = {image_size};
constexpr int kVoteWindow = {vote_window};
constexpr int kClassCount = {len(class_names)};
constexpr int kStreamCount = {len(stream_names)};

const char *kClassNames[kClassCount] = {{
  {", ".join(c_string(name) for name in class_names)}
}};

const char *kStreamNames[kStreamCount] = {{
  {", ".join(c_string(name) for name in stream_names)}
}};

const char kStreamCodes[kStreamCount] = {{
  {", ".join(codes)}
}};

const char *kStreamDisplays[kStreamCount] = {{
  {", ".join(c_string(raw_rules["streams"][name]["display"]) for name in stream_names)}
}};

// Class output index -> stream index. Same order as the model softmax.
const uint8_t kClassToStream[kClassCount] = {{
  {", ".join(str(index) for index in class_to_stream)}
}};

const float kStreamThresholds[kStreamCount] = {{
  {", ".join(f"{float(thresholds[name]):.2f}f" for name in stream_names)}
}};

constexpr char kUnknownCode = '?';
constexpr const char *kUnknownDisplay = "?";
"""
    )


def _read_split(path: Path, split_name: str) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split_name:
                continue
            if Path(row["filepath"]).suffix.lower() not in IMAGE_SUFFIXES:
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


def _stratified_cap(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if len(rows) <= limit:
        return rows
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    rng = np.random.default_rng(seed)
    picked: list[dict] = []
    per_label = max(1, limit // len(by_label))
    for label in sorted(by_label):
        group = by_label[label]
        order = rng.permutation(len(group))
        picked.extend(group[int(index)] for index in order[:per_label])
    return picked[:limit]


if __name__ == "__main__":
    main()
