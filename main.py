import numpy as np
from omegaconf import OmegaConf
import argparse
import os
import matplotlib.pyplot as plt
from MLP import MLP
from Model import EnsembleModel
from Dataset import Dataset
from gpu_utils import set_visible_GPU
from eval_utils import compute_metrics, print_metrics
from constants import *



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

    return parser


def shift_NEP(dataset):
    dataset.add_shifted_series(new_col='shifted_NEP')

def train(cfg, X_train, y_train, scaler):
    model_class = MODELS[cfg.model.model_name]
    trainer_config = cfg.trainer
    model_config = cfg.model
    if cfg.model.get('num_models', 1) > 1 :
        weights = cfg.model.get('weights')
        print("initializing models")
        models = [model_class(model_config=model_config, trainer_config=trainer_config, scaler=scaler) for _ in range(cfg.model.num_models)]
        print("building ensemble model")
        model = EnsembleModel(models, weights)
        print("initialized model")
        resample =cfg.trainer.get("resample_alg", DEFAULT_RESAMPLEING_ALG)
        proportion = cfg.trainer.get("proportion", DEFAULT_RESAMPLING_PROPORTION)
        print("begin training individual models")
        model.train(X_train, y_train, resample, proportion)
    else:
        model = model_class(model_config=model_config, trainer_config=trainer_config, scaler=scaler)
        model.train(X_train, y_train)
    save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "model")
    model.save_model_state(save_path)

def fine_tune(cfg, X_train, y_train):
    model = load_model(cfg, eval=False)
    print("begin fine-tuning")
    resample = cfg.trainer.get("resample_alg", DEFAULT_RESAMPLEING_ALG)
    proportion = cfg.trainer.get("proportion", DEFAULT_RESAMPLING_PROPORTION)
    model.train(X_train, y_train, resample, proportion)
    save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "model")
    model.save_model_state(save_path)

def load_model(cfg, eval=True):
    if eval:
        dir_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name)
    else:
        dir_path = cfg.trainer.get("starting_ckpt_dir", None)
    model_path = os.path.join(dir_path, "model")
    trainer_config = cfg.trainer
    model_class = MODELS[cfg.model.model_name]
    if cfg.model.get('num_models', 1) > 1 :
        models = [model_class() for _ in range(cfg.model.num_models)]
        model = EnsembleModel(models)
    else:
        model = model_class()

    model.load_model_state(model_path)
    model.trainer_config = trainer_config
    return model

def evaluate(cfg, y_hat, y_test):
    metrics = compute_metrics(y_hat, y_test)
    metrics_text = print_metrics(metrics)
    print(metrics_text)
    save_dir = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name)
    metrics_path = os.path.join(save_dir, "metrics.npy")
    metrics_txt_path = os.path.join(save_dir, "metrics.txt")
    np.save(metrics_path, metrics)
    with open(metrics_txt_path, 'w') as f:
        f.write(metrics_text)
    plt.scatter(y_test, y_hat)
    graph_path =  os.path.join(save_dir, f"graph.jpg")    
    plt.savefig(graph_path, format='jpg')
    plt.clf()



MODELS = {
    "MLP": MLP
}

PREPROCESSING = {
    'shifted_NEP': shift_NEP
}


def main_train_single(cfg):
    """
    Train a single model given a configuration.
    This function is used by the ensemble launcher.
    """
    data_path = os.path.join(cfg.data.data_dir, cfg.data.date_str)
    scale_path = os.path.join(cfg.data.data_dir, cfg.data.get("scaler_date", '⚖️'))
    features = cfg.data.get("features", None)
    dataset = Dataset(data_path, scale_path, features)
    
    trim_to = cfg.data.get("dataset_max", None)
    if trim_to is not None:
        dataset.trim(int(trim_to))

    prepros_steps = cfg.data.get("preprocessing", [])
    print("Preprocessing")
    for preprocessing_step in prepros_steps:
        print(f"running {preprocessing_step}")
        PREPROCESSING[preprocessing_step](dataset)

    print("splitting data")
    split_on = cfg.data.get("split_on", "cell_num")
    test_split = [str(id) for id in cfg.data.test_split.split(',')]
    X_train, X_test, y_train, y_test, scaler = dataset.get_scaled_split(test_split, split_on)

    save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name)
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_path, "config.yaml"))

    # Note: GPU selection is handled by the ensemble launcher via CUDA_VISIBLE_DEVICES
    # so we don't call set_visible_GPU here

    if cfg.trainer.get("starting_ckpt_dir", False):
        fine_tune(cfg, X_train, y_train)
    else:
        print("begin training")
        train(cfg, X_train, y_train, scaler)


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    assert args.train + args.eval >= 1, "must select at least one of train or eval should be picked! You have both or are missing both."

    cfg = OmegaConf.load(args.config)

    data_path = os.path.join(cfg.data.data_dir, cfg.data.date_str)
    scale_path = os.path.join(cfg.data.data_dir, cfg.data.get("scaler_date", '⚖️')) #scale_path should error if used when not set
    features = cfg.data.get("features", None)
    dataset = Dataset(data_path, scale_path, features)
    
    trim_to = cfg.data.get("dataset_max", None)
    if trim_to is not None:
        dataset.trim(int(trim_to))

    prepros_steps = cfg.data.get("preprocessing", [])
    print("Preprocessing")
    for preprocessing_step in prepros_steps:
        print(f"running {preprocessing_step}")
        PREPROCESSING[preprocessing_step](dataset)

    print("splitting data")
    split_on = cfg.data.get("split_on", "cell_num")
    test_split = [str(id) for id in cfg.data.test_split.split(',')]
    X_train, X_test, y_train, y_test, scaler = dataset.get_scaled_split(test_split, split_on)

    save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name)
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_path, "config.yaml"))

    set_visible_GPU(cfg)

    if args.train == True:
        if cfg.trainer.get("starting_ckpt_dir", False):
            fine_tune(cfg, X_train, y_train)
        else:
            print("begin training")
            train(cfg, X_train, y_train, scaler)
    
    if args.eval == True:
        model = load_model(cfg)
        print("begin evaluating")
        y_hat = model.predict(X_test)
        evaluate(cfg, y_hat, y_test)
