from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
import datetime
import os
from Model import *
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from typing import Optional
from typing import Optional, Union

DEFAULT_HIDDEN_DIMS = 256
DEFAULT_ACTIVATIONS = [tf.nn.relu, tf.nn.relu, tf.nn.leaky_relu]
DEFAULT_INPUT_DIMS = 5
DEFAULT_OUTPUT_DIMS = 1
DEFAULT_PINN_WEIGHT = 0.0  # Default weight for PINN loss term

class MLP(Model):
    

    def __init__(self, config, scaler: RobustScaler = None):
        # print("in init")
        super().__init__(config, scaler)
        model_cfg = self.config.model
        hidden_dims = model_cfg.get("hidden_dims", DEFAULT_HIDDEN_DIMS)
        activations = model_cfg.get("activations", [tf.nn.relu, tf.nn.leaky_relu, tf.nn.leaky_relu])
        input_dim = model_cfg.get("input_dim", DEFAULT_INPUT_DIMS)
        output_dim = model_cfg.get("output_dim", DEFAULT_OUTPUT_DIMS)

        if type(hidden_dims) is int:
            hidden_dims = [5] + [hidden_dims for _ in range(2)] #in current implementation, first hidden layer is always 5, and the rest are the same size as hidden_dims. Mistake?
        
        assert len(hidden_dims) == len(activations), f"hidden_dims and activations must have the same length. {len(hidden_dims) != len(activations)}."

        # print("about to make layers")
        linear_layers = [
            layers.Dense(hidden_dims[0], input_dim=input_dim, kernel_initializer='normal', bias_initializer=tf.keras.initializers.HeNormal(), activation=activations[0])
        ]
        for hidden_dim, activation in zip(hidden_dims[1:], activations[1:]):
            linear_layers.append(layers.Dense(units=hidden_dim, activation=activation, bias_initializer=tf.keras.initializers.HeNormal(), kernel_initializer='normal'))
        
        linear_layers.append(layers.Dense(units=output_dim, activation="linear", dtype=tf.float32))
        # print("about to make sequential")
        self.model = Sequential(linear_layers)
        # for v in self.model.variables:
            # print(v.name, v.device)
    
    def AV_decrease_loss(self, av_prev, av_pred):
        """Physics-Informed Neural Network (PINN) loss term."""
        return tf.nn.relu(av_pred - av_prev)


    def train(self, X_data, y_data, verbose: bool = True, GPU: int = None):
        """
        Wrapper that chooses an idle GPU (when GPU == "auto") and then
        delegates to the internal _train_impl method.  All model building
        and optimisation occur under the selected device scope.
        """
        if GPU is not None:
            print(f"Training model with GPU {GPU} as specified")
            with tf.device(f"/GPU:{GPU}"):
                return self._train_impl(X_data, y_data, verbose)
            
        return self._train_impl(X_data, y_data, verbose)
    # ------------------------------------------------------------------ #
    # Original training logic moved here (unchanged)                     #
    # ------------------------------------------------------------------ #
    def _train_impl(self, X_data, y_data, verbose=False):
        """
        Train the model with the provided data.

        Args:
            X_data: Input features (numpy array or tf.Tensor).
            y_data: Target values.
            scaler: Scaler used to unscale X_data for PINN loss (required if pinn_weight > 0).
            log_dir: Directory for TensorBoard logs. If None, a default will be used.
            **kwargs: Additional arguments for extensibility.
        """
        #if scaler has not been set during initialization, set it during 
        if self.scaler is None:
            self.scaler = RobustScaler()
            self.scaler = self.scaler.fit(X_data)
            X_data = self.scaler.transform(X_data)
            print("WARNING: Scaler not set during initialization. Assuming unscaled data passed during training!")
        
        pinn_weight = self.config.trainer.get("pinn_weight", DEFAULT_PINN_WEIGHT)

        log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        file_writer = tf.summary.create_file_writer(log_dir)

        batch_size = self.config.trainer.get("batch_size", DEFAULT_BATCH_SIZE)
        num_epochs = self.config.trainer.get("epochs", DEFAULT_EPOCHS)

        if pinn_weight == 0.0:
            # Standard Keras fit with TensorBoard logging
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=2000)
            self.model.compile(
                loss="mse",
                optimizer=self.optimizer,
                metrics=["mse", "mape"]
            )
            self.model.fit(
                X_data,
                y_data,
                epochs=num_epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=1,
                callbacks=[tensorboard_callback]
            )
            # After training, ensure self.optimizer is the trained optimizer state
            self.optimizer = self.model.optimizer
        else:
            # Custom training loop with modular loss (MSE + PINN)
            if self.scaler is None:
                raise ValueError("Scaler must be provided for PINN loss training.")

            mse_loss = tf.keras.losses.MeanSquaredError()

            dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
            dataset = dataset.shuffle(buffer_size=25000).batch(batch_size)

            step = 0
            for epoch in range(num_epochs):
                print(f"epoch: {epoch}")
                for x_batch, y_batch in tqdm(dataset):
                    with tf.GradientTape() as tape:
                        outputs = self.model(x_batch, training=True)
                        mse = mse_loss(y_batch, outputs)

                        # Unscale input to get av_prev (last feature)
                        unscaled_batch = self.scaler.inverse_transform(x_batch.numpy())
                        input_av = tf.expand_dims(tf.cast(unscaled_batch[:, -1], tf.float32), axis=1)
                        pinn = tf.reduce_mean(self.AV_decrease_loss(input_av, outputs))

                        # Modular loss: can add more priors here
                        loss = mse + pinn_weight * pinn

                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    # Logging
                    if step % 1000 == 0:
                        with file_writer.as_default():
                            tf.summary.scalar('total_loss', loss, step=step)
                            tf.summary.scalar('mse_loss', mse, step=step)
                            tf.summary.scalar('pinn_loss', pinn, step=step)
                            tf.summary.scalar('diff', tf.reduce_mean(input_av - outputs), step=step)
                    step += 1

                # Optionally, add validation or test metrics here

        # End of training
        if verbose:
            print("Training complete. Logs saved to:", log_dir)

    def predict(self, X_data, GPU: Optional[Union[int, None]] = None):
        """
        Predict with optional explicit GPU assignment.

        GPU parameter behaviour:
        • int   – specific GPU index
        • "auto" (default) – automatically pick an idle GPU
        • None  – defer to TensorFlow's default placement
        """
        X_data = self.scaler.transform(X_data)
        return super().predict(X_data, GPU).flatten()
