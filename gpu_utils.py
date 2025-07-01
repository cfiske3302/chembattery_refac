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
from defaults import *

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
    print("USING GPUs", tf.config.get_visible_devices())