"""MobileNetV2 width 0.35 at 96x96, plus a classification head.

The backbone is called with training=False in both fine-tuning phases.
BatchNorm then keeps the running statistics it learned on ImageNet. Updating
those statistics on a few thousand images is the usual silent transfer-learning
failure. BatchNorm layers are also left non-trainable so gamma and beta stay put.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


def build_model(
    num_classes: int,
    image_size: int = 96,
    alpha: float = 0.35,
    dropout: float = 0.2,
    weights: str | None = "imagenet",
) -> tuple[tf.keras.Model, tf.keras.Model]:
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        alpha=alpha,
        include_top=False,
        weights=weights,
    )
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    # Constant training=False: fit() cannot flip the backbone into train mode.
    features = backbone(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    dropped = tf.keras.layers.Dropout(dropout)(pooled)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(dropped)
    model = tf.keras.Model(inputs, outputs)
    return model, backbone


def set_backbone_trainable(backbone: tf.keras.Model, trainable_fraction: float) -> int:
    """Unfreeze the top fraction of backbone layers. 0 freezes the backbone.

    Returns the number of backbone layers left trainable. BatchNorm stays frozen
    either way; the forward call also passes training=False.
    """
    layers = list(backbone.layers)
    if trainable_fraction <= 0:
        backbone.trainable = False
    else:
        backbone.trainable = True
        n_freeze = int(round(len(layers) * (1.0 - trainable_fraction)))
        n_freeze = min(max(n_freeze, 0), len(layers))
        for layer in layers[:n_freeze]:
            layer.trainable = False
    for layer in layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return sum(1 for layer in layers if layer.trainable)


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def trainable_parameter_count(model: tf.keras.Model) -> int:
    return int(sum(int(tf.size(weight)) for weight in model.trainable_weights))
