import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax
from typing import Callable, Union
import numpy as np

def off_mae_loss(use_mask: bool = False):
    """
    Combined MAE loss function with Huber loss component and optional masking.

    Args:
        use_mask (bool): If True, applies masking for non-zero values in y_true.
                        If False, calculates loss on all values.

    Returns:
        function: Loss function that takes y_true and y_pred tensors
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Huber loss component for robustness
        huber_delta = 1.0
        error = y_pred - y_true
        abs_error = K.abs(error)
        quadratic = K.minimum(abs_error, huber_delta)
        linear = abs_error - quadratic
        loss_per_pixel = 0.5 * quadratic ** 2 + huber_delta * linear

        # Calculate mean across all dimensions except batch
        axis_to_reduce = range(1, K.ndim(y_pred))

        if use_mask:
            # Create mask for non-zero values
            w = K.cast(K.not_equal(y_true, 0), 'float32')
            # Apply mask and calculate weighted mean
            num = K.sum(w, axis=axis_to_reduce) + K.epsilon()
            masked_loss = K.sum(w * loss_per_pixel, axis=axis_to_reduce)
            return masked_loss / num
        else:
            # Calculate simple mean without masking
            return K.mean(loss_per_pixel, axis=axis_to_reduce)

    return loss

def off_mse_loss(use_mask: bool = False):
    """
    Combined MSE loss function with optional masking.

    Args:
        use_mask (bool): If True, applies masking for non-zero values in y_true.
                        If False, calculates loss on all values.

    Returns:
        function: Loss function that takes y_true and y_pred tensors
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Calculate squared error
        squared_error = K.square(y_pred - y_true)

        # Calculate mean across all dimensions except batch
        axis_to_reduce = range(1, K.ndim(y_pred))

        if use_mask:
            # Create mask for non-zero values
            w = K.cast(K.not_equal(y_true, 0), 'float')
            # Apply mask and calculate weighted mean
            num = K.sum(w, axis=axis_to_reduce) + K.epsilon()
            mse = K.sum(w * squared_error, axis=axis_to_reduce)
            return mse / num / 2
        else:
            # Calculate simple mean without masking
            return K.mean(squared_error, axis=axis_to_reduce) / 2

    return loss
