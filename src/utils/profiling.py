"""
Model profiling and resource measurement utilities.

Provides functions to count trainable parameters, measure inference
latency with GPU synchronization, track RAM usage, and time arbitrary
function calls. Used in Section 12 (Efficiency Analysis).
"""

import time
from typing import Any, Callable, Dict

import numpy as np
import psutil
import tensorflow as tf


def count_trainable_params(model: tf.keras.Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


def get_ram_usage_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def _sync_if_needed() -> None:
    # Force materialization/synchronization of pending GPU work in eager mode
    # by converting a tiny tensor result to numpy.
    if tf.config.list_physical_devices("GPU"):
        tf.constant(0.).numpy()


def measure_inference_latency(
    model: tf.keras.Model,
    sample_input: Any,
    n_warmup: int = 10,
    n_runs: int = 30,
) -> Dict[str, float]:
    for _ in range(n_warmup):
        y = model(sample_input, training=False)
        if hasattr(y, "numpy"):
            y.numpy()
        _sync_if_needed()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        y = model(sample_input, training=False)
        if hasattr(y, "numpy"):
            y.numpy()
        _sync_if_needed()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times, dtype=float)
    batch_size = int(sample_input.shape[0]) if hasattr(sample_input, "shape") else 1

    return {
        "latency_mean_ms": float(times.mean() * 1000),
        "latency_std_ms": float(times.std() * 1000),
        "latency_p50_ms": float(np.percentile(times, 50) * 1000),
        "latency_p95_ms": float(np.percentile(times, 95) * 1000),
        "throughput_samples_s": float(batch_size / times.mean()),
    }


def timed_call(fn: Callable, *args, **kwargs) -> Dict[str, Any]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return {
        "result": result,
        "elapsed_seconds": end - start,
    }
