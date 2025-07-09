from abc import ABC, abstractmethod
import os
import pickle
import numpy as np
from typing import List, Iterable
import tensorflow as tf
import multiprocessing as mp
from sklearn.preprocessing import RobustScaler
from constants import *

OPTIMIZERS = {
    "Adam": tf.keras.optimizers.legacy.Adam,
    "SGD": tf.keras.optimizers.legacy.SGD,
}
# @dataclass
# class ModelConfig:
#     learning_rate: float
#     batch_size: int
#     num_epochs: int
#     optimizer: str = "adam"
#     #for adam
#     beta_1: float=0.9
#     beta_2: float=0.999
#     #for SGD
#     momentum: float=0.0

class Model(ABC):
    def __init__(self, model_config=None, trainer_config=None, scaler: RobustScaler = None): 
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.model = None

        if trainer_config is None:
            self.optimizer = None
        else:
            self._create_optimizer()   
        
        self.scaler = scaler

    def _create_optimizer(self):
        opt = self.trainer_config.get("optimizer", DEFAULT_OPTIMIZER)
        if opt=="adam":
            self.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.trainer_config.get("learning_rate", DEFAULT_LEARNING_RATE),
                beta_1=self.trainer_config.get("beta_1", DEFAULT_BETA_1),
                beta_2=self.trainer_config.get("beta_2", DEFAULT_BETA_2)
            )
        elif opt=="SGD":
            self.optimizer = tf.keras.optimizers.legacy.SGD(
                learning_rate=self.trainer_config.get("learning_rate", DEFAULT_LEARNING_RATE),
                momentum=self.trainer_config.get("momentum", DEFAULT_MOMENTUM)
            )
        elif opt is None:
            self.optimizer = None
            raise Warning("No optimizer specified, this could be correct if using a model without GD")
        else:
            raise ValueError(f"Unrecognized optimizer {opt}")

    def reset_train_config(self, trainer_config):
        self.trainer_config = trainer_config
        self._create_optimizer()

    @abstractmethod
    def train(self, X_data, y_data, verbose=False, GPU=None):
        """Train the model with the provided data."""
        pass

    @abstractmethod
    def predict(self, X_data, GPU=None):
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X_data : array-like
        GPU : int | str | None
            • int  – specific GPU index  
            • "auto" (default) – automatically pick an idle GPU  
            • None – let TensorFlow choose the device.
        """
        # Resolve GPU automatically, if requested
        if GPU is not None:
            with tf.device(f"/GPU:{GPU}"):
                predictions = self.model.predict(X_data)
        else:
            predictions = self.model.predict(X_data)
        return predictions
    
    def save(self, path):
        self.model.save(os.path.join(path, "model"))
    
    def load(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, "model"))

    def save_model_state(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save Keras model
        self.model.save(os.path.join(path, "model"))
        # Save the rest of the instance (excluding self.model)
        state = self.__dict__.copy()
        state.pop("model", None)
        state.pop("optimizer", None)
        # print(state)
        with open(os.path.join(path, "instance.pkl"), "wb") as f:
            pickle.dump(state, f)
        # Save optimizer state if present
        if self.optimizer is not None:
            optimizer_weights = self.optimizer.get_weights()
            with open(os.path.join(path, "optimizer.pkl"), "wb") as f:
                pickle.dump(optimizer_weights, f)

    def load_model_state(self, path: str, skip_optimizer: bool = False):
        # Load Keras model
        self.model = tf.keras.models.load_model(os.path.join(path, "model"))
        # Load the rest of the instance
        with open(os.path.join(path, "instance.pkl"), "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        self._create_optimizer()
        # Load optimizer state if present
        optimizer_path = os.path.join(path, "optimizer.pkl")
        if self.optimizer is not None and os.path.exists(optimizer_path) and not skip_optimizer:
            with open(optimizer_path, "rb") as f:
                optimizer_weights = pickle.load(f)

            # Ensure all optimizer slot variables exist before restoring
            if hasattr(self.optimizer, "build"):
                # Public API available in newer TF versions
                self.optimizer.build(self.model.trainable_variables)
            else:
                # Fallback: run a single dummy update to create slots
                zero_grads = [tf.zeros_like(v) for v in self.model.trainable_variables]
                self.optimizer.apply_gradients(
                    zip(zero_grads, self.model.trainable_variables),
                    experimental_aggregate_gradients=False,
                )
            self.optimizer.set_weights(optimizer_weights)


class EnsembleModel:
    def __init__(self, models: List[Model], weights: List[float] = None):
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights

            
    def subsample_data(self, X, y, proportion):
        rows = np.random.choice(X.shape[0], size = int(X.shape[0]*proportion)) # Randomly choose 70% of data
        return X[rows,:], y[rows]

    def reset_train_config( self, trainer_config):
        """
        Reset the training configuration for all models in the ensemble.
        """
        for model in self.models:
            model.reset_train_config(trainer_config)

    def bootstrap_data(self, X_data, y_data, proportion):
        if not (0 < proportion < 1):
            raise ValueError("Proportion must be between 0 and 1.")
        n_samples = int(len(X_data) * proportion)

        # Generate random indices with replacement
        if isinstance(X_data, tf.Tensor):
            indices = tf.random.uniform(
                shape=(n_samples,),
                maxval=len(X_data),
                dtype=tf.int32,
            )
            X_bootstrap = tf.gather(X_data, indices)
            y_bootstrap = tf.gather(y_data, indices)
        else:
            # NumPy-based sampling for array-like or list-like objects
            indices = np.random.randint(0, len(X_data), size=n_samples)
            try:
                # Works for NumPy arrays, pandas Series/DataFrame, etc.
                X_bootstrap = X_data[indices]
                y_bootstrap = y_data[indices]
            except (TypeError, IndexError):
                # Fallback for pure Python sequences
                X_bootstrap = [X_data[i] for i in indices]
                y_bootstrap = [y_data[i] for i in indices]
        return X_bootstrap, y_bootstrap
    
    # @staticmethod
    # def worker(model_path_queue, X_data, y_data, gpu_idx):
    #     while True:
    #         try:
    #             model_path = model_path_queue.get(timeout=3)  # Get a model from the queue
    #             model = Model()  # Create a new model instance
    #             model.load(model_path)  # Load the model from the path
    #         except mp.queues.Empty:
    #             break
    #         with tf.device(f'/GPU:{gpu_idx}'):
    #             model.train(X_data, y_data)

    def train(self, X_data, y_data, resample="subsample", proportion: float = 0.7,
              available_gpus: Iterable[int] = None,
              **kwargs):
        """
        Train each model in the ensemble, distributing them across available GPUs if specified.

        Args:
            X_data: Input features.
            y_data: Target values.
            available_gpus: List of GPU indices to use for training. If None, trains sequentially on default device.
            resample: one of:
                subsample: randomly subsample the data for each model
                full: use the full dataset for each model
                bootstrap: use bootstrap sampling for each model
                custom: use a custom sampling function provided in kwargs
            proportion: proportion of data to use for subsampling (if resample is "subsample" or "bootstrap").
        """
        if resample == "subsample":
            resample = lambda X, y: self.subsample_data(X, y, proportion)
        elif resample == "bootstrap":
            resample = lambda X, y: self.bootstrap_data(X, y, proportion)
        elif resample == "full":
            resample = lambda X, y: (X, y)
        elif resample == "custom":
            if "custom_sampling_func" not in kwargs:
                raise ValueError("Custom sampling function must be provided in kwargs.")
            resample = kwargs["custom_sampling_func"]
        else:
            raise ValueError(f"Unrecognized resampling method: {resample}")
        

        if available_gpus is None or len(available_gpus) == 1:
            # Only one GPU, train sequentially on that GPU
            print(f"GPU passed to ensemble is {available_gpus}")
            gpu_idx = None if available_gpus is None else available_gpus[0]
            for model in self.models:
                model.train(*resample(X_data, y_data), GPU=gpu_idx, verbose=True)     
        elif len(available_gpus) > 1:
            # TODO
            raise NotImplementedError("Parallel training across multiple GPUs is not yet implemented.")
            # mp.set_start_method("spawn")  # safer on Linux and required on Windows/Mac

            # # Create a queue with all the models to train
            # task_queue = mp.Queue()
            # for model in self.models:
            #     task_queue.put(model)

            # processes = []
            # for gpu_id in available_gpus:
            #     p = mp.Process(target=worker, args=(task_queue, X_data, y_data, gpu_id))
            #     p.start()
            #     processes.append(p)

            # # Wait for all processes to finish
            # for p in processes:
            #     p.join()

            # print("All models have been trained.")       
    
    def predict(self, X_data, GPU=None):
        predictions = [model.predict(X_data, GPU) for model in self.models]
        weighted_preds = sum(w * p for w, p in zip(self.weights, predictions))
        return weighted_preds
    
    def save_model_state(self, path: str):
        """
        Save the ensemble state and all constituent models.

        Directory structure created:

        path/
            ensemble_state.pkl     # pickled dict with all ensemble attrs except the models themselves
            models/
                model_0/
                model_1/
                ...
        """
        # Create root and models directory
        os.makedirs(path, exist_ok=True)
        models_dir = os.path.join(path, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Pickle all ensemble attributes except the list of models themselves
        state = self.__dict__.copy()
        state.pop("models")  # Exclude models from the state
        with open(os.path.join(path, "ensemble_state.pkl"), "wb") as f:
            pickle.dump(state, f)

        # Save each individual model
        for idx, model in enumerate(self.models):
            model_path = os.path.join(models_dir, f"model_{idx}")
            model.save_model_state(model_path)

    def load_model_state(self, path: str, skip_optimizer: bool=False):
        """
        Load the ensemble state and all constituent models from *path*.
        Note: self.models must already contain instantiated Model objects
        in the same order and of the same length as when saved.
        """
        # Restore ensemble-level attributes
        state_path = os.path.join(path, "ensemble_state.pkl")
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            # Update attributes except models
            for k, v in state.items():
                setattr(self, k, v)

        # Restore each sub-model
        models_dir = os.path.join(path, "models")
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f"Expected models directory not found: {models_dir}")

        saved_subdirs = sorted(
            d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))
        )
        if len(saved_subdirs) != len(self.models):
            raise ValueError(
                f"Mismatch between saved models ({len(saved_subdirs)}) and current ensemble "
                f"({len(self.models)})."
            )

        for idx, model in enumerate(self.models):
            model_path = os.path.join(models_dir, f"model_{idx}")
            model.load_model_state(model_path, skip_optimizer=skip_optimizer)    