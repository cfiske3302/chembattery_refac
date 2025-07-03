#!/usr/bin/env python3
"""
Example script demonstrating how to use the multiprocessing ensemble launcher.

This script shows three ways to train multiple models concurrently:

1. Using the standalone ensemble_launcher.py script
2. Using the EnsembleModel class with multiple GPUs
3. Using the new GPU memory estimation functions
"""

from gpu_utils import get_gpu_free_mem, pick_gpu, estimate_peak_mb
from omegaconf import OmegaConf

def example_gpu_memory_check():
    """Example of checking GPU memory and estimating model requirements."""
    print("=== GPU Memory Check Example ===")
    
    # Check free memory on all GPUs
    free_memory = get_gpu_free_mem()
    print(f"Free GPU memory: {free_memory} MB")
    
    # Example model configuration
    model_cfg = {
        "input_dim": 5,
        "hidden_dims": [5, 256, 256],
        "output_dim": 1
    }
    trainer_cfg = {
        "batch_size": 1024
    }
    
    # Estimate memory requirements
    estimated_mb = estimate_peak_mb(model_cfg, trainer_cfg)
    print(f"Estimated memory requirement: {estimated_mb} MB")
    
    # Find a suitable GPU
    gpu_idx = pick_gpu(estimated_mb)
    if gpu_idx is not None:
        print(f"GPU {gpu_idx} has sufficient memory")
    else:
        print("No GPU has sufficient memory")

def example_ensemble_launcher_usage():
    """Example of using the standalone ensemble launcher."""
    print("\n=== Ensemble Launcher Usage ===")
    print("To use the standalone ensemble launcher:")
    print("python ensemble_launcher.py config1.yaml config2.yaml config3.yaml")
    print("\nOptional arguments:")
    print("--memory-safety-factor 1.5  # Multiply memory estimates by this factor")

def example_ensemble_model_usage():
    """Example of using EnsembleModel with multiple GPUs."""
    print("\n=== EnsembleModel Multi-GPU Usage ===")
    print("In your training script:")
    print("""
from MLP import MLP
from Model import EnsembleModel

# Create multiple models
models = [MLP(model_config, trainer_config, scaler) for _ in range(4)]

# Create ensemble
ensemble = EnsembleModel(models)

# Train across multiple GPUs (will use memory-aware scheduling)
available_gpus = [0, 1]  # Use GPUs 0 and 1
ensemble.train(X_train, y_train, 
               resample="bootstrap", 
               proportion=0.8,
               available_gpus=available_gpus)
""")

if __name__ == "__main__":
    example_gpu_memory_check()
    example_ensemble_launcher_usage()
    example_ensemble_model_usage()
    
    print("\n=== Summary ===")
    print("The multiprocessing dispatcher provides:")
    print("• Memory-aware GPU scheduling")
    print("• Automatic load balancing across GPUs")
    print("• Streaming data with tf.data.Dataset")
    print("• GPU memory growth to prevent OOM errors")
    print("• Support for both standalone launcher and EnsembleModel")
