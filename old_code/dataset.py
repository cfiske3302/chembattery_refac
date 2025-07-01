import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
import os
import numpy as np

from .data_utils import train_test_split, add_shifted_series

def load_dataset(date_str, data_dir):
    xCC = pd.read_pickle(os.path.join(data_dir, date_str, 'xCC.pkl'))              #pd.read_pickle(os.path.join(data_dir, date_str, 'xCC.pkl'))
    yCC = pd.read_pickle(os.path.join(data_dir, date_str, 'yCC.pkl'))

    # xCC_amend = add_shifted_series(xCC, yCC)

    # xCC_amend.to_pickle("./output_csv_amend.pkl")

    with open(os.path.join(data_dir, date_str, 'masterList.pkl'), 'rb') as f:
        masterList = pickle.load(f)

    return xCC, yCC, masterList

def traintest_split(xCC, yCC, test_split, master_list):
    return train_test_split(xCC, yCC, 'cell', test_split, master_list)

def filter_features(xCC, features):
    return xCC[features]

def scale_data(xCC, data_dir, scaler_date, scaler=None):
    input_scaler = scaler if scaler is not None else RobustScaler()

    if "shifted_NEP" in xCC.columns:
        xCC["shifted_NEP"] += np.random.normal(0, 0.1)

    if scaler is None:
        input_scaler = input_scaler.fit(xCC)

    xCC_scaled = input_scaler.transform(xCC)
    # with open(os.path.join(data_dir, scaler_date, "scaler.pkl"), 'rb') as f:
    #     input_scaler = pickle.load(f)

    # if input_scaler.n_features_in_ != xCC.shape[1]:
    #     scaleX = xCC[:,:input_scaler.n_features_in_]
    #     xCC_scaled = np.concat(input_scaler.transform(xCC), xCC[:,input_scaler.n_features_in_:], axis=1)
    # else:
    #     xCC_scaled = input_scaler.transform(xCC)
    return xCC_scaled, input_scaler

def scale_data_df(xCC, data_dir, scaler_date, scaler=None):
    input_scaler = scaler if scaler is not None else RobustScaler()

    if scaler is None:
        input_scaler = input_scaler.fit(xCC)

    xCC_scaled = pd.DataFrame(
        input_scaler.transform(xCC),
        columns=xCC.columns,
        index=xCC.index,
    )

    return xCC_scaled, input_scaler

def get_scaled_split_cycle_data(date_str, data_dir, test_split, scaler_date, features):
    xCC, yCC, masterlist = load_dataset(date_str, data_dir)

    xCC = filter_features(xCC, features)

    xCC_train, yCC_train, xCC_test, yCC_test = traintest_split(xCC, yCC, test_split, masterlist)

    xCC_train, scaler = scale_data(xCC_train, data_dir, scaler_date)
    xCC_test, _ = scale_data_df(xCC_test, data_dir, scaler_date, scaler=scaler)

    return xCC_train, xCC_test, np.array(yCC_train), yCC_test, scaler

def get_scaled_split_data(date_str, data_dir, test_split, scaler_date, features):
    xCC, yCC, masterlist = load_dataset(date_str, data_dir)

    xCC = filter_features(xCC, features)

    xCC_train, yCC_train, xCC_test, yCC_test = traintest_split(xCC, yCC, test_split, masterlist)

    xCC_train, scaler = scale_data(xCC_train, data_dir, scaler_date)
    xCC_test, _ = scale_data(xCC_test, data_dir, scaler_date, scaler=scaler)

    return xCC_train, xCC_test, np.array(yCC_train), np.array(yCC_test), scaler
