import numpy as np
import tensorflow as tf
from positional_encodings.tf_encodings import TFPositionalEncoding1D
from einops import repeat

def pos_enc(x, L=10):
    b, _ = x.shape
    x = tf.expand_dims(x, axis=-1)
    x = repeat(x, 'b x 1 -> b x c', c=L)

    pos_enc = TFPositionalEncoding1D(L)

    return pos_enc(x).reshape(b, -1)

def tune_last_layer_only(model):
    for l in range(len(model.layers)-1):
        model.layers[l].trainable = False

    return model

def calculate_r_squared(y_true, y_pred):
    """
    Calculate R-squared (coefficient of determination) for regression.
    
    Parameters:
    y_true : array-like, true values
    y_pred : array-like, predicted values
    
    Returns:
    r_squared : float, R-squared value
    """
    # Calculate the mean of the true values
    mean_true = np.mean(y_true)
    
    # Calculate the total sum of squares (TSS)
    total_sum_squares = np.sum((y_true - mean_true)**2)
    
    # Calculate the residual sum of squares (RSS)
    residual_sum_squares = np.sum((y_true - y_pred)**2)
    
    # Calculate R-squared
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    
    return r_squared