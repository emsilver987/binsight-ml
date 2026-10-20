"""Manifest rows to a tf.data pipeline.

Augmentation lives here, on the training split only. The model itself is just
MobileNetV2 plus a head, which is what TFLite conversion needs later.
Val and test take the deployment path: center-crop, resize to 96x96, RGB,
then mobilenet_v2.preprocess_input.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from src.util import rows_for_split


def make_dataset(
    rows: list[dict],
    image_root,
    class_to_index: dict[str, int],
    image_size: int,
    batch_size: int,
    training: bool,
    augment: dict,
    seed: int,
) -> tf.data.Dataset:
    if not rows:
        raise ValueError("cannot build a dataset from zero rows")

    paths = [str(image_root / row["filepath"]) for row in rows]
    labels = [class_to_index[row["label"]] for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    # tf.image.random_* is frozen inside a dataset map once the global seed is set,
    # so training draws from an explicit generator whose state advances every sample.
    rng = tf.random.Generator.from_seed(seed, alg="philox") if training else None
    loader = _loader(image_size, training, augment, rng)
    dataset = dataset.map(loader, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.deterministic = not training
    return dataset.with_options(options)


def class_weights(rows: list[dict], class_names: list[str]) -> dict[int, float]:
    """Inverse-frequency weights. TrashNet's trash class is ~137 images."""
    counts = np.zeros(len(class_names), dtype=np.float64)
    index = {name: i for i, name in enumerate(class_names)}
    for row in rows:
        counts[index[row["label"]]] += 1
    missing = [class_names[i] for i, count in enumerate(counts) if count == 0]
    if missing:
        raise ValueError(f"training split has no rows for: {missing}")
    total = float(counts.sum())
    n_classes = len(class_names)
    return {i: total / (n_classes * float(counts[i])) for i in range(n_classes)}


def batches_of(split_items: dict[str, list[str]], items: dict[str, dict]) -> dict[str, list[dict]]:
    return {name: rows_for_split(items, ids) for name, ids in split_items.items()}


def _loader(image_size: int, training: bool, augment: dict, rng):
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    margin = int(augment["crop_margin"])
    max_radians = float(augment["rotation_degrees"]) * np.pi / 180.0
    brightness_delta = float(augment["brightness_delta"]) * 255.0
    contrast_lower = float(augment["contrast_lower"])
    contrast_upper = float(augment["contrast_upper"])
    horizontal_flip = bool(augment["horizontal_flip"])

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = _center_crop_square(image)
        image = tf.cast(image, tf.float32)
        if training:
            side = image_size + margin
            image = tf.image.resize(image, [side, side])
            image = _rotate(image, rng.uniform([], -max_radians, max_radians))
            offset_limit = side - image_size
            offset_y = rng.uniform([], 0, offset_limit + 1, dtype=tf.int32)
            offset_x = rng.uniform([], 0, offset_limit + 1, dtype=tf.int32)
            image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, image_size, image_size)
            if horizontal_flip:
                image = tf.cond(
                    rng.uniform([]) < 0.5,
                    lambda: tf.image.flip_left_right(image),
                    lambda: image,
                )
            image = image + rng.uniform([], -brightness_delta, brightness_delta)
            factor = rng.uniform([], contrast_lower, contrast_upper)
            mean = tf.reduce_mean(image)
            image = (image - mean) * factor + mean
        else:
            image = tf.image.resize(image, [image_size, image_size])
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = preprocess(image)
        return image, tf.cast(label, tf.int32)

    return load


def _center_crop_square(image):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    side = tf.minimum(height, width)
    offset_height = (height - side) // 2
    offset_width = (width - side) // 2
    return tf.image.crop_to_bounding_box(image, offset_height, offset_width, side, side)


def _rotate(image, angle):
    """Rotate inside the data pipeline, not as a layer in the model.

    ImageProjectiveTransformV3 samples the source with the inverse map, so a
    positive angle rotates the content counterclockwise.
    """
    image = tf.cast(image, tf.float32)
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    transform = tf.stack(
        [
            cos_a,
            sin_a,
            center_x - cos_a * center_x - sin_a * center_y,
            -sin_a,
            cos_a,
            center_y + sin_a * center_x - cos_a * center_y,
            0.0,
            0.0,
        ]
    )
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transform[tf.newaxis, ...],
        output_shape=tf.shape(image)[:2],
        fill_mode="REFLECT",
        interpolation="BILINEAR",
        fill_value=0.0,
    )
    return rotated[0]
