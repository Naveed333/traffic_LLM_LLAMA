"""Programmatic computation of MAE and MSE metrics."""

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


def compute_mae(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> float:
    """Compute Mean Absolute Error (MAE).

    Args:
        y_true: Ground truth values (24-element array/list).
        y_pred: Predicted values (24-element array/list).

    Returns:
        MAE as float.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length ({len(y_true)}) != y_pred length ({len(y_pred)})"
        )

    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mse(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> float:
    """Compute Mean Squared Error (MSE).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MSE as float.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length ({len(y_true)}) != y_pred length ({len(y_pred)})"
        )

    return float(np.mean((y_true - y_pred) ** 2))


def mae_converged(mae_new: float, mae_old: float, threshold: float = 0.001) -> bool:
    """Check if MAE has converged (change is below threshold).

    Args:
        mae_new: MAE of current iteration.
        mae_old: MAE of previous iteration.
        threshold: Convergence threshold.

    Returns:
        True if converged, False otherwise.
    """
    return abs(mae_new - mae_old) < threshold
