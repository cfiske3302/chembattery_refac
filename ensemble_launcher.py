#!/usr/bin/env python3
"""
Multiprocessing dispatcher for training multiple models concurrently on GPUs.

Usage:
    python ensemble_launcher.py config1.yaml config2.yaml config3.yaml

This script:
1. Estimates the peak GPU memory requirement for each model
2. Polls available GPUs for free memory
3. Launches training processes only when sufficient GPU memory is available
4. Uses multiprocessing to run models in parallel across available GPUs
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from typing import List

import tensorflow as tf
from omegaconf import OmegaConf

from gpu_utils import get_gpu_free_mem, pick_gpu, estimate_peak_mb
from main import main_train_single  # We'll need to refactor main.py to expose this


def enable_memory_growth():
    """
    Enable GPU memory growth to prevent TensorFlow from pre-allocating
    the entire GPU memory at startup.
    """
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"Warning: Could not set memory growth for {gpu}: {e}")


def train_single_model(config_path: str, gpu_idx: int):
    """
    Train a single model on the specified GPU.
    
    This function runs in a separate process and sets CUDA_VISIBLE_DEVICES
    so the child process only sees the assigned GPU.
    """
    # Set environment variable so this process only sees the assigned GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    
    # Enable memory growth in the child process
    enable_memory_growth()
    
    print(f"Process {os.getpid()}: Training {config_path} on GPU {gpu_idx}")
    
    try:
        # Load config and train
        cfg = OmegaConf.load(config_path)
        main_train_single(cfg)
        print(f"Process {os.getpid()}: Completed training {config_path}")
    except Exception as e:
        print(f"Process {os.getpid()}: Error training {config_path}: {e}")
        raise


def launch_ensemble(config_paths: List[str], memory_safety_factor: float = 1.5):
    """
    Launch multiple training jobs across available GPUs.
    
    Parameters
    ----------
    config_paths : List[str]
        Paths to YAML configuration files for each model to train
    memory_safety_factor : float
        Multiply estimated memory requirements by this factor for safety
    """
    # Enable memory growth in the main process
    enable_memory_growth()
    
    # Calculate memory requirements for each config
    memory_requirements = []
    for config_path in config_paths:
        cfg = OmegaConf.load(config_path)
        estimated_mb = estimate_peak_mb(cfg.model, cfg.trainer)
        safe_mb = int(estimated_mb * memory_safety_factor)
        memory_requirements.append(safe_mb)
        print(f"Config {config_path}: estimated {estimated_mb} MB, using {safe_mb} MB with safety factor")
    
    # Queue of (config_path, memory_needed) tuples
    pending_jobs = list(zip(config_paths, memory_requirements))
    active_processes = []
    
    print(f"Starting ensemble launcher with {len(pending_jobs)} jobs")
    
    while pending_jobs or active_processes:
        # Clean up finished processes
        active_processes = [p for p in active_processes if p.is_alive()]
        
        # Try to launch new jobs
        jobs_to_remove = []
        for i, (config_path, memory_needed) in enumerate(pending_jobs):
            gpu_idx = pick_gpu(memory_needed)
            if gpu_idx is not None:
                print(f"Launching {config_path} on GPU {gpu_idx} (needs {memory_needed} MB)")
                process = mp.Process(
                    target=train_single_model,
                    args=(config_path, gpu_idx)
                )
                process.start()
                active_processes.append(process)
                jobs_to_remove.append(i)
            else:
                free_mem = get_gpu_free_mem()
                print(f"Waiting for GPU with {memory_needed} MB free. Current free memory: {free_mem}")
        
        # Remove launched jobs from pending queue
        for i in reversed(jobs_to_remove):
            pending_jobs.pop(i)
        
        if pending_jobs:
            print(f"Waiting 10 seconds... {len(pending_jobs)} jobs pending, {len(active_processes)} active")
            time.sleep(10)
        elif active_processes:
            print(f"All jobs launched. Waiting for {len(active_processes)} processes to complete...")
            time.sleep(5)
    
    print("All training jobs completed!")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Launch multiple model training jobs across GPUs"
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="YAML configuration files for models to train"
    )
    parser.add_argument(
        "--memory-safety-factor",
        type=float,
        default=1.5,
        help="Multiply estimated memory requirements by this factor (default: 1.5)"
    )
    return parser


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = get_parser()
    args = parser.parse_args()
    
    # Validate config files exist
    for config_path in args.configs:
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
    
    print(f"Ensemble launcher starting with configs: {args.configs}")
    launch_ensemble(args.configs, args.memory_safety_factor)
