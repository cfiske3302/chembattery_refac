from Model import EnsembleModel
# Test training LinearModel on 2024-08-21 data
import pickle
import os
import numpy as np
from old_code.dataset import get_scaled_split_data
from MLP import MLP
from Model import ModelConfig, Model

if __name__=="__main__":
    date_str = "2024-08-21"
    data_dir = "../Features/"
    test_split = list(range(1,53))
    scaler_date = "2024-08-21"

    # Try to load feature list from xCC.pkl
    features = [  "Energy","Capacity","Voltage-8","Energy-8","Power-8"]

    # Load data
    X_train, X_test, y_train, y_test, scaler = get_scaled_split_data(
        date_str, data_dir, test_split, scaler_date, features
    )



    config = ModelConfig(
        optimizer= 'adam',
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=1
    )

    models = [MLP(config) for _ in range(5)]
    em = EnsembleModel(models)
    em.train(X_train, y_train, available_gpus=[0,1,2])