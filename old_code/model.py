import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import keras
from sklearn.metrics import mean_squared_error
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
from omegaconf import OmegaConf
import pickle
from sklearn.preprocessing import RobustScaler

from .model_utils import calculate_r_squared

def load_from_folder(exp_name):
    # Load config
    config_path = os.path.join("./runs", exp_name, "config.yaml")
    cfg = OmegaConf.load(config_path)

    # Get number of models from config
    num_models = cfg.trainer.num_models if cfg.trainer.ensemble else 1

    # Load models
    models = []
    for i in range(num_models):
        model_path = os.path.join("./runs", exp_name, "models", f"model_{i}", "model.keras")
        if os.path.exists(model_path):
            models.append(tf.keras.models.load_model(model_path))

    # Load scaler if exists
    scaler = None
    scaler_path = os.path.join("./runs", exp_name, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    return models, scaler

def linear_model_builder_compile(input_dim, hidden_dim, output_dim, learning_rate):
    model = Sequential()

    model.add(layers.Dense(5, input_dim = input_dim, kernel_initializer = 'normal', bias_initializer= tf.keras.initializers.HeNormal,  activation = tf.nn.relu))
    model.add(layers.Dense(units = hidden_dim, activation = tf.nn.relu, bias_initializer=tf.keras.initializers.HeNormal, kernel_initializer = 'normal'))
    model.add(layers.Dense(units = hidden_dim, activation = tf.nn.leaky_relu, bias_initializer=tf.keras.initializers.HeNormal, kernel_initializer = 'normal'))

    # if(num_models == 1):
    #     model.add(layers.Dense(48,activation = tf.nn.leaky_relu, bias_initializer=tf.keras.initializers.HeNormal, dtype=tf.float32))
    
    model.add(layers.Dense(output_dim, activation = 'linear', dtype=tf.float32))

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.001)
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse', 'mape'])

    return model

def linear_model_builder(input_dim, hidden_dim, output_dim):
    model = Sequential()

    model.add(layers.Dense(5, input_dim = input_dim, kernel_initializer = 'normal', bias_initializer= tf.keras.initializers.HeNormal,  activation = tf.nn.relu))
    model.add(layers.Dense(units = hidden_dim, activation = tf.nn.relu, bias_initializer=tf.keras.initializers.HeNormal, kernel_initializer = 'normal'))
    model.add(layers.Dense(units = hidden_dim, activation = tf.nn.leaky_relu, bias_initializer=tf.keras.initializers.HeNormal, kernel_initializer = 'normal'))

    # if(num_models == 1):
    #     model.add(layers.Dense(48,activation = tf.nn.leaky_relu, bias_initializer=tf.keras.initializers.HeNormal, dtype=tf.float32))
    
    model.add(layers.Dense(output_dim, activation='linear', dtype=tf.float32))

    return model

def predict_test(xCC_test, yCC_test, model, scaler, initial_est_model, est_scaler):
    # Create empty dataframe with same structure as yCC_test
    predictions = yCC_test.copy()
    predictions.iloc[:] = 0

    # Iterate through each cell
    for cell_num in xCC_test.index.get_level_values('cell_num').unique():
        print(f"Predicting Cell: {cell_num}")
        cell_data = xCC_test.xs(cell_num, level='cell_num')
        
        # Get all cycles for this cell
        cycles = cell_data.index.get_level_values('cycle').unique()
        print(f"There are {len(cycles)} cycles")
        # Find max length of cycles
        max_len = max(len(xCC_test.loc[(cell_num, cycle)]) for cycle in cycles)
        
        # Initialize arrays for batched prediction
        batch_features = []
        cycle_lengths = []
        initial_voltages = []
        
        # Prepare data for each cycle
        for cycle in cycles:
            # print(yCC_test.loc[(cell_num, cycle)])
            cycle_data = xCC_test.loc[(cell_num, cycle)]
            cycle_lengths.append(len(cycle_data))
            # print(cycle_data)
            # print(cycle_data)
            # Pad features if needed
            features = cycle_data.values[:,:-1]  # All features except voltage
            if len(features) < max_len:
                # print(features, features.shape)
                pad_length = max_len - len(features)
                features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
            batch_features.append(features)
            
            # Get initial prediction from initial estimation model
            if initial_est_model is not None:
                initial_features = cycle_data.iloc[0].values  # First timestep features
                initial_features_unscaled = scaler.inverse_transform(initial_features[None,:])
                initial_features_est_scaled = est_scaler.transform(initial_features_unscaled[:,:-1])
                initial_pred = initial_est_model(initial_features_est_scaled, training=False)
                # print(initial_pred)
                initial_voltages.append(tf.reshape(initial_pred, []).numpy())
            else:
                initial_voltages.append(cycle_data.iloc[0].values[-1])
            # print(len(batch_features), batch_features[0].shape, len(initial_voltages), initial_voltages)
            # print("---------")

        batch_features = np.array(batch_features)
        
        # Initialize predictions for all cycles using initial estimates
        batch_preds = np.expand_dims(np.array(initial_voltages), axis=1)

        # print(batch_features.shape)
        # prnt(batch_preds.shape)
        gt_vals = yCC_test.loc[(cell_num, cycle)].values
        
        # Predict each timestep for all cycles simultaneously
        for t in tqdm(range(max_len)):

            # Create input batch with last predictions
            # print("****************")
            current_features = batch_features[:,t,:]
            # print(current_features)
            # Need to scale previous predictions before using as input
            # Create full input array to scale properly
            unscaled_input = np.zeros((batch_preds.shape[0], current_features.shape[1] + 1))
            unscaled_input[:,:-1] = scaler.inverse_transform(np.concatenate((current_features, np.zeros((current_features.shape[0], 1))), axis=1))[:,:-1]  # Unscale features first
            # print(yCC_test.loc[(cell_num, cycle)])

            unscaled_input[:,-1] = gt_vals[t] if t < len(gt_vals) else 0#batch_preds[:,-1]  # Add previous predictions
            # Now scale the full input
            # print("-----")
            # print(unscaled_input)
            scaled_input = scaler.transform(unscaled_input)
            # print(scaled_input)
            # Get model predictions (which are already unscaled)
            preds = model(scaled_input, training=False)
            # print(preds)
            batch_preds = np.column_stack((batch_preds, tf.reshape(preds, [-1]).numpy()))
            # print(qq)
            # print(batch_preds)
        # Store predictions for each cycle (removing padding)
        for cycle, length, cycle_preds in zip(cycles, cycle_lengths, batch_preds):
            predictions.loc[(cell_num, cycle)] = cycle_preds[1:length+1]  # Remove initial value and padding
        
    return predictions

def eval_test(xCC_test, yCC_test, model):
    ind = xCC_test.index
    results = defaultdict(list)
    for cell_num, group1 in xCC_test.groupby(level=0):
        print(f"Eval Cell: {cell_num}")
        for cycle_num, group2 in group1.groupby(level=1):
            print(group1.shape)
            results[cell_num].append(eval_cycle(group2.values[:,:-1], group2.values[:,1], model))
            # print(f"  Level 1: {name2}")
            # print(f"  Data: {group2.values}")
    return results

def eval_cycle(cycle_feat, cycle_av, model):
    print(cycle_feat.shape)
    model_pred = [cycle_av[0]]

    for t in tqdm(range(cycle_feat.shape[0])):
        av_pred = model(np.append(cycle_feat[t], model_pred[-1])[None,:], training=False)
        model_pred.append(tf.reshape(av_pred, []).numpy())

    return np.array(model_pred)

def compute_metrics(y_pred, y_gt):
    mpe = abs(y_pred - y_gt)
    ae = (y_pred-y_gt)
    mae = mpe.mean()
    me = ae.mean()

    rmse = mean_squared_error(y_gt, y_pred, squared=False)
    rmspe = np.sqrt(np.mean(np.square(((y_gt - y_pred) / abs(y_gt))), axis=0)) * 100
    percent_rmse = (rmse/np.mean(y_gt))*100
    percent_rmse2 = (rmse/(max(y_gt)-min(y_gt)))*100

    r2 = calculate_r_squared(y_pred, y_gt)

    return np.array([mae, me, rmse, percent_rmse, rmspe, r2])
