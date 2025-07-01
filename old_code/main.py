import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from omegaconf import OmegaConf
import argparse
import os
import importlib
import shutil
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import pickle
from .dataset import get_scaled_split_data, get_scaled_split_cycle_data
from .model import linear_model_builder, compute_metrics, eval_test, predict_test, load_from_folder, linear_model_builder_compile
from .model_utils import pos_enc
import itertools
import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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

def get_obj_from_str(string, reload=True):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def AV_decrease_loss(av_prev, av_pred):
    return tf.nn.relu(av_pred - av_prev)

def train(cfg):
    if cfg.trainer.ensemble:
        count = cfg.trainer.num_models
    else:
        count = 1
    
    test_split = [int(x) for x in cfg.data.test_split.split(",")]
    dataset = get_scaled_split_cycle_data(cfg.data.date_str, cfg.data.data_dir, test_split, cfg.data.scaler_date, cfg.data.features)

    x_train, x_test, y_train, y_test, scaler = dataset

    scaler_save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "scaler.pkl")

    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)

    if cfg.trainer.pos_enc:
        x_train = pos_enc(x_train)
        x_test = pos_enc(x_test)

    mse_loss = tf.keras.losses.MeanSquaredError()

    # if cfg.trainer.initial_est_model is not None:
    #     initial_est_model, est_scaler = load_from_folder(cfg.trainer.initial_est_model)
    # else:
    #     initial_est_model = None
    #     est_scaler = None

    with tf.device("/gpu:0"):
        preds = []
        for i in range(count):
            print(f"Training ensemble model {i+1} of {count}")
            rows = np.random.choice(x_train.shape[0], size = int(x_train.shape[0]*0.7)) # Randomly choose 70% of data

            x_train_sample = x_train[rows,:]
            y_train_sample = y_train[rows]

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train_sample, y_train_sample))
            train_dataset = train_dataset.shuffle(buffer_size=25000).batch(cfg.trainer.batch_size)

            model = linear_model_builder(cfg.model.input_dim, cfg.model.hidden_dim, cfg.model.output_dim)
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=cfg.trainer.learning_rate)

            log_dir = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "logs", f"logs_{i}")
            file_writer = tf.summary.create_file_writer(log_dir)

            model_step = 0
            os.makedirs(os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "models"), exist_ok=True)
            for epoch in tqdm(range(cfg.trainer.epochs)):
                for x_batch, y_batch in tqdm(train_dataset):
                    with tf.GradientTape() as tape:
                        print("x_batch: ", x_batch.shape)
                        outputs = model(x_batch, training=True)
                        print("outputs: ", outputs.shape)
                        mse = mse_loss(y_batch, outputs)

                        unscaled_batch = scaler.inverse_transform(x_batch)
                        input_av = tf.expand_dims(tf.cast(unscaled_batch[:,-1], tf.float32), axis=1)
                        print("input_av: ", input_av.shape)
                        pinn = tf.reduce_mean(AV_decrease_loss(input_av, outputs))
                        loss = 10 * mse + 0.1 * pinn

                        gradients = tape.gradient(loss, model.trainable_variables)

                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                        if model_step % 20 == 0:
                            with file_writer.as_default():
                                tf.summary.scalar('combined_loss', loss, step=model_step)
                                tf.summary.scalar('mse_loss', mse, step=model_step)
                                tf.summary.scalar('pinn_loss', pinn, step=model_step)
                                tf.summary.scalar('diff', tf.reduce_mean(input_av - outputs), step=model_step)

                        model_step += 1

                if epoch % 10 == 0:
                    y_pred = np.array(predict_test(x_test, y_test, model, scaler, initial_est_model[0], est_scaler))
                    metrics = compute_metrics(y_pred, y_test)
                    mae, me, rmse, percent_rmse, rmspe, r2 = metrics
                    with file_writer.as_default():
                        tf.summary.scalar('mae', mae, step=epoch * len(train_dataset))
                        tf.summary.scalar('me', me, step=epoch * len(train_dataset))
                        tf.summary.scalar('rmse', rmse, step=epoch * len(train_dataset))
                        tf.summary.scalar('percent_rmse', percent_rmse, step=epoch * len(train_dataset))
                        tf.summary.scalar('rmspe', rmspe, step=epoch * len(train_dataset))
                        tf.summary.scalar('r2', r2, step=epoch * len(train_dataset))

            save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "models", f"model_{i}")
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, "model.keras")
            model.save(save_path)

            y_pred = np.array(predict_test(x_test, y_test, model, scaler, initial_est_model[0], est_scaler))

            # y_pred = model.predict(x_test).flatten()
            preds.append(y_pred)

    ensemble_pred = np.stack(preds).mean(axis=0)

    np.save(os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "pred.npy"), ensemble_pred)

    metrics = compute_metrics(ensemble_pred, y_test)

    plt.scatter(y_test, ensemble_pred)
    plt.savefig(os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "pred.jpg"))

    save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "metrics.npy")    
    np.save(save_path, metrics)

    return metrics

def train_fit(cfg):
    if cfg.trainer.ensemble:
        count = cfg.trainer.num_models
    else:
        count = 1
    
    test_split = [int(x) for x in cfg.data.test_split.split(",")]
    dataset = get_scaled_split_data(cfg.data.date_str, cfg.data.data_dir, test_split, cfg.data.scaler_date, cfg.data.features)

    x_train, x_test, y_train, y_test, _ = dataset

    if cfg.trainer.dataset_max  > 0:
        dataset_size = x_train.shape[0]
    else:
        dataset_size = int(cfg.trainer.dataset_max)

    x_train = x_train[:dataset_size]

    if cfg.trainer.pos_enc:
        x_train = pos_enc(x_train)
        x_test = pos_enc(x_test)

    preds = []
    for i in range(count):
        model = linear_model_builder_compile(cfg.model.input_dim, cfg.model.hidden_dim, cfg.model.output_dim, cfg.trainer.learning_rate)

        rows = np.random.choice(x_train.shape[0], size = int(x_train.shape[0]*0.7)) # Randomly choose 70% of data

        x_train_sample = x_train[rows,:]
        y_train_sample = y_train[rows]

        log_dir = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "logs", f"logs_{i}")

        model.fit(x_train_sample, y_train_sample, epochs=cfg.trainer.epochs, \
                   batch_size=cfg.trainer.batch_size, shuffle=True, verbose=1, \
                    callbacks=[TensorBoard(log_dir, update_freq=200)])

        save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "models", f"model_{i}")
        os.makedirs(save_path, exist_ok=True)
        model.save(filepath=save_path)

        y_pred = model.predict(x_test).flatten()
        preds.append(y_pred)

    ensemble_pred = np.stack(preds).mean(axis=0)

    metrics = compute_metrics(ensemble_pred, y_test)

    save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "metrics.npy")    
    np.save(save_path, metrics)

    return metrics

def finetune(cfg):
    if cfg.trainer.ensemble:
        count = cfg.trainer.num_models
    else:
        count = 1
    
    test_split = [int(x) for x in cfg.data.test_split.split(",")]
    dataset = get_scaled_split_data(cfg.data.date_str, cfg.data.data_dir, test_split, cfg.data.scaler_date, cfg.data.features)

    x_train, x_test, y_train, y_test = dataset

    if cfg.trainer.pos_enc:
        x_train = pos_enc(x_train)
        x_test = pos_enc(x_test)

    if cfg.model.ft_mod:
        ft_mod_fn = get_obj_from_str(cfg.model.ft_mod)
    else:
        ft_mod_fn = lambda x: x

    preds = []
    for i in range(count):
        model = ft_mod_fn(tf.keras.models.load_model(os.path.join(cfg.trainer.ckpt_dir, f"model_{i}")))

        rows = np.random.choice(x_train.shape[0], size = int(x_train.shape[0]*0.7)) # Randomly choose 70% of data

        x_train_sample = x_train[rows,:]
        y_train_sample = y_train[rows]

        model.fit(x_train_sample, y_train_sample, epochs=cfg.trainer.epochs, batch_size=cfg.trainer.batch_size, shuffle=True, verbose=1)

        save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "models", f"model_{i}")
        os.makedirs(save_path, exist_ok=True)
        model.save(filepath=save_path)

        y_pred = model.predict(x_test).flatten()
        preds.append(y_pred)

    ensemble_pred = np.stack(preds).mean(axis=0)

    metrics = compute_metrics(ensemble_pred, y_test)

    save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "metrics.npy")    
    np.save(save_path, metrics)

    return metrics

def eval_model_cycle(cfg):
    if cfg.trainer.ensemble:
        count = cfg.trainer.num_models
    else:
        count = 1
    
    test_split = [int(x) for x in cfg.data.test_split.split(",")]
    dataset = get_scaled_split_cycle_data(cfg.data.date_str, cfg.data.data_dir, test_split, cfg.data.scaler_date, cfg.data.features)

    x_train, x_test, y_train, y_test, scaler = dataset

    print(y_test.shape)

    scaler_save_path = os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "scaler.pkl")

    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)

    if cfg.trainer.pos_enc:
        x_train = pos_enc(x_train)
        x_test = pos_enc(x_test)

    mse_loss = tf.keras.losses.MeanSquaredError()

    if cfg.trainer.initial_est_model is not None:
        initial_est_model, est_scaler = load_from_folder(cfg.trainer.initial_est_model)
    else:
        initial_est_model = None
        est_scaler = None

    preds = []
    for i in range(1):
        model = tf.keras.saving.load_model(os.path.join(cfg.trainer.ckpt_dir, f"model_{i}", "model.keras"))

        y_pred = predict_test(x_test, y_test, model, scaler, initial_est_model[0], est_scaler)

        for cell_num in y_pred.index.get_level_values('cell_num').unique():
            print(f"Predicting Cell: {cell_num}")
            cell_data = y_pred.xs(cell_num, level='cell_num')
            
            # Get all cycles for this cell
            cycles = cell_data.index.get_level_values('cycle').unique()
            for cycle in cycles:
                pred_data = y_pred.loc[(cell_num, cycle)]
                gt_data = y_test.loc[(cell_num, cycle)]
                x_test_data = x_test.loc[(cell_num, cycle)]
                # fig, ax = plt.subplots(2)
                plt.plot(np.array(gt_data), label="GT")
                plt.plot(np.array(pred_data), label="Pred")
                plt.legend()
                # ax[0,2].plot(np.array(x_test_data["Energy"][:300]))
                # ax[0,3].plot(np.array(x_test_data["Capacity"][:300]))
                # ax[1,0].plot(np.array(x_test_data["Voltage-8"][:300]))
                # ax[1,1].plot(np.array(x_test_data["Energy-8"][:300]))
                # ax[1,2].plot(np.array(x_test_data["Power-8"][:300]))
                plt.savefig(f"pred_{cell_num}_{cycle}.jpg")
                plt.close()
                

        preds.append(y_pred)

    print(preds[0])
    print(preds[0].shape)
    print(y_test)

    ensemble_pred = np.stack(preds).mean(axis=0)

    print(ensemble_pred.shape)

    metrics = compute_metrics(ensemble_pred, y_test)
    print(metrics)
    # save_path = os.path.join(cfg.trainer.save_dir, "models", cfg.data.date_dir, cfg.trainer.exp_name, "metrics.npy")    
    # np.save(save_path, metrics)

    plt.plot(y_test, ensemble_pred)
    plt.show()

    return metrics


def eval_model(cfg):
    if cfg.trainer.ensemble:
        count = cfg.trainer.num_models
    else:
        count = 1
    
    test_split = [int(x) for x in cfg.data.test_split.split(",")]
    dataset = get_scaled_split_data(cfg.data.date_str, cfg.data.data_dir, test_split, cfg.data.scaler_date, cfg.data.features)

    x_train, x_test, y_train, y_test, scaler = dataset

    if cfg.trainer.pos_enc:
        x_train = pos_enc(x_train)
        x_test = pos_enc(x_test)

    preds = []
    for i in range(count):
        model = tf.keras.models.load_model(os.path.join(cfg.trainer.ckpt_dir, f"model_{i}"))

        y_pred = model.predict(x_test).flatten()
        preds.append(y_pred)

    ensemble_pred = np.stack(preds).mean(axis=0)

    metrics = compute_metrics(ensemble_pred, y_test)
    print(metrics)
    root_dir = os.path.join(cfg.trainer.save_dir, "models", cfg.data.date_str, cfg.trainer.exp_name)
    os.makedirs(root_dir, exist_ok=True)
    save_path = os.path.join(root_dir, "metrics.npy")    
    np.save(save_path, metrics)

    plt.plot(y_test, ensemble_pred)
    plt.show()

    return metrics

def eval_individual_base_learners(cfg):
    if cfg.trainer.ensemble:
        count = cfg.trainer.num_models
    else:
        count = 1
    
    test_split = [int(x) for x in cfg.data.test_split.split(",")]
    dataset = get_scaled_split_data(cfg.data.date_str, cfg.data.data_dir, test_split, cfg.data.scaler_date, cfg.data.features)

    x_train, x_test, y_train, y_test, scaler = dataset

    if cfg.trainer.pos_enc:
        x_train = pos_enc(x_train)
        x_test = pos_enc(x_test)
    
    root_dir = os.path.join(cfg.trainer.save_dir, "evaluations", cfg.data.date_str, cfg.trainer.exp_name)
    os.makedirs(root_dir, exist_ok=True)
    model_names = []
    preds = []
    for i in range(count):
        print(f"evaluating model {i}")
        save_path =  os.path.join(root_dir, f"preds_for_model_{i}.npy")
        if os.path.exists(save_path):
            print(f"found existing output for model {i}, skipping eval")
            y_pred = np.load(save_path)
        else:
            model = tf.keras.models.load_model(os.path.join(cfg.trainer.ckpt_dir, f"model_{i}"))
            y_pred = model.predict(x_test).flatten()
            np.save(save_path, y_pred)
        
        model_names.append(str(i))
        preds.append(y_pred)
        np.save(save_path, y_pred)

    preds = np.array(preds)
    model_names = np.array(model_names)
    indicies = list(range(5))

    results = []
    for r in range(1, 6):  # r is the length of each combination
        for combination in itertools.combinations(indicies, r):
            print(f"evaluating models {combination}")
            idx = [*combination]
            curr_preds = preds[idx]
            # print(curr_preds.shape)
            curr_model_names = model_names[idx]

            ensemble_pred = np.stack(curr_preds).mean(axis=0)
            # print(y_pred.shape)
            # print(ensemble_pred.shape)
            metrics = compute_metrics(ensemble_pred, y_test)
            mae, me, rmse, percent_rmse, rmspe, r2 = metrics

            # save_path =  os.path.join(root_dir, f"metrics_for_models_{''.join(curr_model_names)}.npy")    
            # np.save(save_path, metrics)
            results.append({
                "model": "".join([str(i) for i in idx]),
                "MAE": mae,
                "Mean Error": me,
                "RMSE": rmse,
                "% RMSE": percent_rmse,
                "RMSPE": rmspe,
                "R²": r2
            })

            plt.scatter(y_test, ensemble_pred)
            save_path =  os.path.join(root_dir, f"graph_for_models_{''.join(curr_model_names)}.jpg")    
            plt.savefig(save_path, format='jpg')
            plt.clf()
    
    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by length of model ID then numerically
    df["model_sort_key"] = df["model"].apply(lambda x: (len(x), int(x)))
    df = df.sort_values(by="model_sort_key").drop(columns="model_sort_key").reset_index(drop=True)

    save_path =  os.path.join(root_dir, f"metrics.csv")
    df.to_csv(save_path)

    return metrics


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    assert args.train + args.eval + args.eval_base_learners == 1, "Exactly one of train, eval, or eval_base_learners should be picked! You have both or are missing both."

    cfg = OmegaConf.load(args.config)
    
    try:
        os.makedirs(cfg.trainer.save_dir + cfg.trainer.exp_name)
    except:
        print("That exp name already exists!")

    shutil.copy2(args.config, os.path.join(cfg.trainer.save_dir, cfg.trainer.exp_name, "config.yaml"))



    if args.train:
        if not 'ckpt_dir' in cfg.trainer:
            train_fit(cfg)
        else:
            finetune(cfg)
    elif args.eval_base_learners:
        eval_individual_base_learners(cfg)
    else:
        eval_model(cfg)
