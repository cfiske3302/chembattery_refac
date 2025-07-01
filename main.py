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
from defaults import *



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
    if cfg.model.get('num_models', 1) > 1 :
        weights = cfg.model.get('weights')
        print("initializing models")
        models = [model_class(cfg, scaler=scaler) for _ in range(cfg.model.num_models)]
        print("building ensemble model")
        model = EnsembleModel(models, weights)
        print("initialized model")
        resample =cfg.trainer.get("resample_alg", DEFAULT_RESAMPLEING_ALG)
        proportion = cfg.trainer.get("proportion", DEFAULT_RESAMPLING_PROPORTION)
        print("begin training individual models")
        model.train(X_train, y_train, resample, proportion)
    else:
        model = model_class(cfg, scaler=scaler)
        model.train(X_train, y_train)
    save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "model")
    model.save_model_state(save_path)

def load_model(cfg):
    dir_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name)
    model_path = os.path.join(dir_path, "model")
    model_config = OmegaConf.load(os.path.join(dir_path, "config.yaml"))
    model_class = MODELS[model_config.model.model_name]
    if cfg.model.get('num_models', 1) > 1 :
        models = [model_class(model_config, scaler=scaler) for _ in range(cfg.model.num_models)]
        model = EnsembleModel(models)
        model.load_model_state(model_path)
    else:
        model = model_class(cfg)
        model.load_model_state(model_path)
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


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()

    assert args.train + args.eval == 1, "Exactly one of train, eval, or eval_base_learners should be picked! You have both or are missing both."

    cfg = OmegaConf.load(args.config)

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

    set_visible_GPU(cfg)

    if args.eval == True:
        model = load_model(cfg)
        print("begin evaluating")
        y_hat = model.predict(X_test)
        evaluate(cfg, y_hat, y_test)
    if args.train == True:
        print("begin training")
        train(cfg, X_train, y_train, scaler)

    
