"""
Environment and reproducibility utilities.

Provides functions to set global random seeds across all libraries,
query GPU/CPU device information, and configure TensorFlow GPU memory.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_device_info() -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    return {
        "tensorflow_version": tf.__version__,
        "num_gpus": len(gpus),
        "gpus": [gpu.name for gpu in gpus],
        "num_cpus": len(cpus),
        "cpus": [cpu.name for cpu in cpus],
    }


def enable_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


if __name__ == "__main__":
    set_global_seed(42)
    enable_gpu_memory_growth()
    print(get_device_info())
