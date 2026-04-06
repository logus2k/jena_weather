import tensorflow as tf
import numpy as np


def compute_saliency_map(model, input_sample, forecast_step=None):
    """
    input_sample: shape (1, lookback, n_features)
    forecast_step: int or None
        - if None, uses mean over forecast horizon
        - if int, explains that specific forecast output step
    """
    x = tf.convert_to_tensor(input_sample, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x, training=False)

        if forecast_step is None:
            target = tf.reduce_mean(y_pred)
        else:
            target = y_pred[:, forecast_step]

    grads = tape.gradient(target, x)
    saliency = tf.abs(grads).numpy()[0]  # (lookback, n_features)

    return saliency


def aggregate_saliency_over_time(saliency):
    return np.mean(saliency, axis=0)


def aggregate_saliency_over_features(saliency):
    return np.mean(saliency, axis=1)
