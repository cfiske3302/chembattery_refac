from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import pickle
import numpy as np
from typing import List
import tensorflow as tf
import multiprocessing as mp


@dataclass
class ModelConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int
    optimizer: str = "adam"
    #for adam
    beta_1: float=0.9
    beta_2: float=0.999
    #for SGD
    momentum: float=0.0
    

OPTIMIZERS = {
    "Adam": tf.keras.optimizers.legacy.Adam,
    "SGD": tf.keras.optimizers.legacy.SGD,
}

class Model(ABC):
    def __init__(self, ModelConfig: ModelConfig): 
        self.config = ModelConfig
        self.model = None
        self._create_optimizer()
        print(self.optimizer)

    def _create_optimizer(self):
        opt = self.config.optimizer
        if opt=="adam":
            self.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.config.learning_rate,
                beta_1=self.config.beta_1,
                beta_2=self.config.beta_2
            )
        elif opt=="SGD":
            self.optimizer = tf.keras.optimizers.legacy.SGD(
                learning_rate=self.config.learning_rate,
                momentum=self.config.momentum
            )
        elif opt is None:
            self.optimizer = None
            raise Warning("No optimizer specified, this could be correct if using a model without GD")
        else:
            raise ValueError(f"Unrecognized optimizer {opt}")


    @abstractmethod
    def train(self, X_data, y_data, verbose=False):
        """Train the model with the provided data."""
        pass

    @abstractmethod
    def predict(self, X_data):
        """Make predictions using the trained model."""
        return self.model.predict(X_data)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save Keras model
        self.model.save(os.path.join(path, "model"))
        # Save the rest of the instance (excluding self.model)
        state = self.__dict__.copy()
        state.pop("model", None)
        state.pop("optimizer", None)
        print(state)
        with open(os.path.join(path, "instance.pkl"), "wb") as f:
            pickle.dump(state, f)
        # Save optimizer state if present
        if self.optimizer is not None:
            optimizer_weights = self.optimizer.get_weights()
            with open(os.path.join(path, "optimizer.pkl"), "wb") as f:
                pickle.dump(optimizer_weights, f)

    def load(self, path: str):
        # Load Keras model
        self.model = tf.keras.models.load_model(os.path.join(path, "model"))
        # Load the rest of the instance
        with open(os.path.join(path, "instance.pkl"), "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        self._create_optimizer()
        # Load optimizer state if present
        optimizer_path = os.path.join(path, "optimizer.pkl")
        if self.optimizer is not None and os.path.exists(optimizer_path):
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

    def subsample_data(self, X_data, y_data, proportion):
        # Subsample the data based on the given proportion
        if not (0 < proportion < 1):
            raise ValueError("Proportion must be between 0 and 1.")
        num_samples = int(len(X_data) * proportion)
        return X_data[:num_samples], y_data[:num_samples]
    
    def bootstrap_data(self, X_data, y_data, proportion):
        """
        Bootstrap sample the data based on the given proportion.

        Args:
            X_data: Input features.
            y_data: Target values.
            proportion: Fraction of the full dataset to draw (0 < proportion < 1).

        Returns:
            A tuple (X_bootstrap, y_bootstrap) containing samples drawn
            WITH replacement from the original data, of size
            int(len(X_data) * proportion).
        """
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

    def train(self, X_data, y_data, resample="subsample", proportion: float=0.7, available_gpus: List[int]=None, **kwargs):
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

        
        if available_gpus is None:
            for model in self.models:
                model.train(X_data, y_data)
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

        elif len(available_gpus) == 1:
            # Only one GPU, train sequentially on that GPU
            gpu_idx = available_gpus[0]
            for model in self.models:
                with tf.device(f'/GPU:{gpu_idx}'):
                    model.train(X_data, y_data)            

    def predict(self, X_data):
        predictions = [model.predict(X_data) for model in self.models]
        weighted_preds = sum(w * p for w, p in zip(self.weights, predictions))
        return weighted_preds
    
    def save(self, path: str):
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
        state = self.__dict__.items()
        state.pop("models")  # Exclude models from the state
        with open(os.path.join(path, "ensemble_state.pkl"), "wb") as f:
            pickle.dump(state, f)

        # Save each individual model
        for idx, model in enumerate(self.models):
            model_path = os.path.join(models_dir, f"model_{idx}")
            model.save(model_path)

    def load(self, path: str):
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
            model.load(model_path)
