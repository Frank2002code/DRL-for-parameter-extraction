import numpy as np

def calculate_rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Percentage Error (RMSPE).
    RMSPE = sqrt(1/n * sum((y_true - y_pred)^2 / y_true^2))
    """
    # Prevent division by zero for very small currents
    y_true_safe = np.where(np.abs(y_true) < 1e-12, 1e-12, y_true)
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true_safe)))
    return rmspe