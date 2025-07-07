#Not used. Dependancy issues with constants
from abc import ABC, abstractmethod
import os
import pickle
import numpy as np
from typing import List, Iterable
import tensorflow as tf
import multiprocessing as mp
from sklearn.preprocessing import RobustScaler
from constants import *
from omegaconf import OmegaConf


def load_model_from_path(path: str, scaler: RobustScaler = None):
    """
    Load a model from the specified path.
    """
    config = OmegaConf.load(os.path.join(path, "config.yaml"))
    model_class = MODELS[config.model.model_name]
    model = model_class(model_config=config.model, scaler=scaler)
    model.model = tf.keras.models.load_model(os.path.join(path, "model"))
    return model