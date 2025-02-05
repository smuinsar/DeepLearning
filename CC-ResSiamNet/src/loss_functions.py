import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax
from typing import Callable, Union
import numpy as np

def off_mae_loss(use_mask: bool = False, use_huber: bool = True, huber_delta: float = 0.1):
    """
    Combined MAE loss function with optional Huber loss component and masking.
    Args:
        use_mask (bool): If True, applies masking for non-zero values in y_true.
                        If False, calculates loss on all values.
        use_huber (bool): If True, applies Huber loss component for robustness.
                         If False, uses pure MAE loss.
        huber_delta (float): threshold between quadratic and linear regions.
                         for a large error (>huber_delta): MAE-like. for a small error (<huber_delta): MSE-like
    Returns:
        function: Loss function that takes y_true and y_pred tensors
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true

        if use_huber:
            # Huber loss component for robustness
            abs_error = K.abs(error)
            quadratic = K.minimum(abs_error, huber_delta)
            linear = abs_error - quadratic
            loss_per_pixel = 0.5 * quadratic ** 2 + huber_delta * linear
        else:
            # Pure MAE loss
            loss_per_pixel = K.abs(error)

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

def off_mse_loss(use_mask: bool = False, use_huber: bool = False, huber_delta: float = 0.1):
    """
    Combined MSE loss function with optional Huber loss and masking.
    Args:
        use_mask (bool): If True, applies masking for non-zero values in y_true.
                        If False, calculates loss on all values.
        use_huber (bool): If True, applies Huber loss component for robustness.
                         If False, uses pure MSE loss.
        huber_delta (float): threshold between quadratic and linear regions.
                         for a large error (>huber_delta): MAE-like. for a small error (<huber_delta): MSE-like
    Returns:
        function: Loss function that takes y_true and y_pred tensors
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_pred - y_true

        if use_huber:
            # Huber loss component for robustness
            abs_error = K.abs(error)
            quadratic = K.minimum(abs_error, huber_delta)
            linear = abs_error - quadratic
            loss_per_pixel = 0.5 * quadratic ** 2 + huber_delta * linear
        else:
            # Pure MSE loss
            loss_per_pixel = K.square(error) / 2

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
