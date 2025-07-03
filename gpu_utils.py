"""
Utility helpers related to GPU selection for training and inference.

Main export
-----------
pick_idle_gpu()  -- Return the index of an idle GPU (low utilisation and
                    ample free memory) or None if no suitable GPU exists.

Implementation notes
--------------------
* First tries the NVIDIA Management Library via the 'pynvml' package.
* If that fails, falls back to parsing 'nvidia-smi' CLI output.
* All string literals are plain ASCII to avoid encoding issues.
"""

from __future__ import annotations

import subprocess
from typing import Optional

import tensorflow as tf
from constants import *

def pick_idle_gpu() -> Optional[int]:
    """
    Parse the output of 'nvidia-smi' to find an idle GPU.

    Returns
    -------
    int | None
        Index of the selected GPU, or None if parsing fails or no idle GPU.
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    best_idx, best_free = None, 0
    for idx, line in enumerate(output.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            free_mb = int(parts[0])
        except ValueError:
            continue

        if free_mb > best_free:
            best_idx, best_free = idx, free_mb

    return best_idx

def get_gpu_free_mem() -> list[int]:
    """
    Return a list with the free memory (in MiB) for each visible GPU.

    Uses `nvidia-smi --query-gpu=memory.free` so it works even when NVML
    Python bindings (`pynvml`) are not installed.
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        return [int(x) for x in output.strip().splitlines()]
    except Exception:
        # If the command fails (e.g. system without NVIDIA GPU),
        # return an empty list so callers can handle gracefully.
        return []

def estimate_peak_mb(model_cfg: dict, trainer_cfg: dict | None = None) -> int:
    """
    Very rough upper-bound of model footprint on the GPU *including one batch*.

    Parameters
    ----------
    model_cfg : dict
        The "model" sub-section of the Hydra / OmegaConf YAML.
        Expected keys (with defaults):
            input_dim        (int, default 5)
            hidden_dims      (int | list[int], default 256 or [5,256,256])
            output_dim       (int, default 1)
    trainer_cfg : dict | None
        If provided, batch_size is taken from here, otherwise DEFAULT_BATCH_SIZE.
    """
    # ---------------- Weights + optimiser state ----------------
    input_dim = model_cfg.get("input_dim", 5)
    output_dim = model_cfg.get("output_dim", 1)
    hidden_dims = model_cfg.get("hidden_dims", 256)
    if isinstance(hidden_dims, int):
        hidden_dims = [5, hidden_dims, hidden_dims]
    layer_dims = [input_dim] + hidden_dims + [output_dim]

    # param count for fully-connected network  Σ_in×out + biases
    n_params = sum(a * b for a, b in zip(layer_dims[:-1], layer_dims[1:])) + sum(layer_dims[1:])
    # 4×: weights fp32 + grads + optimiser slots (Adam has m & v)
    weights_mb = 4 * n_params * 4 / 1_000_000  # bytes→MiB

    # ---------------- Activations (one batch) ------------------
    batch_size = (trainer_cfg or {}).get("batch_size", DEFAULT_BATCH_SIZE)
    dtype_bytes = 4  # assume fp32 activations
    max_width = max(hidden_dims)
    activ_mb = batch_size * max_width * dtype_bytes * 2 * 1.25 / 1_000_000
    #       ^-- fwd + back-prop  ×  safety factor

    return int(weights_mb + activ_mb + 32)  # +32 MiB cushion

def pick_gpu(memory_needed_mb: int) -> int | None:
    """
    Return the index of a GPU with at least *memory_needed_mb* free.
    If none exists, return None.
    """
    visible_gpus = [gpu.name.split(':')[-1] for gpu in tf.config.list_physical_devices('GPU')]
    free_list = get_gpu_free_mem()
    for idx, free_mb in enumerate(free_list):
        if free_mb >= memory_needed_mb and str(idx) in visible_gpus:
            return idx
    return None

def set_visible_GPU(cfg):
    GPUs = cfg.trainer.get("GPUs", DEFAULT_GPU)
    if GPUs == "auto":
        GPUs = [pick_idle_gpu()]
    elif isinstance(GPUs, int):
        GPUs = [GPUs]
    else:
        GPUs = [int(gpu) for gpu in GPUs.split(',')]
    
    print("using GPUs: ", GPUs)
    all_gpus = tf.config.list_physical_devices('GPU')
    visible_GPUs = [all_gpus[i] for i in GPUs]
    tf.config.set_visible_devices(visible_GPUs, 'GPU')
    # print("USING GPUs", tf.config.get_visible_devices())
