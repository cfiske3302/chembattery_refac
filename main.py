import numpy as np
from omegaconf import OmegaConf
import argparse
import os
import importlib
import shutil
import matplotlib
import matplotlib.pyplot as plt
import pickle
from .dataset import get_scaled_split_data, get_scaled_split_cycle_data
from .model import linear_model_builder, compute_metrics, eval_test, predict_test, load_from_folder, linear_model_builder_compile
from .model_utils import pos_enc
import itertools
import pandas as pd
from MLP import MLP
from Model import EnsembleModel
from Dataset import Dataset

MODELS = {
    "MPL": MLP_builder
}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True
    )
    parser.add_argument(
        "--train",
        action="store_true"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=False
    )
    parser.add_argument(
        "--eval",
        action="store_true",
    )
    parser.add_argument(
        "--eval_base_learners",
        action="store_true",
    )

    return parser

def MLP_builder(parser, cfg):
    if cfg.model.ckpt_dir is not None:
        model_dir = cfg.model.ckpt_dir

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    assert args.train + args.eval + args.eval_base_learners == 1, "Exactly one of train, eval, or eval_base_learners should be picked! You have both or are missing both."

    cfg = OmegaConf.load(args.config)

    model_builder = MODELS[cfg.model.model_name]
    model_builder(parser, cfg)
