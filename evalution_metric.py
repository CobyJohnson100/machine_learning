# machine_learning/evalution_metric.py
from dataclasses import dataclass

import numpy as np

@dataclass
class EvaluationMetrics:
    r2: float
    mse: float
    rss: float
    tss: float

    def __str__(self):
        return (f"R-squared (R²): {self.r2:.4f}\n"
                f"Mean Squared Error (MSE): {self.mse:.4f}\n"
                f"Residual Sum of Squares (RSS): {self.rss:.4f}\n"
                f"Total Sum of Squares (TSS): {self.tss:.4f}")
    
def calculate_evaluation_metrics(y_actual: np.ndarray, y_predict: np.ndarray):
    tss = np.sum((y_actual - np.mean(y_actual)) ** 2) # Calculate total sum of squares (TSS)
    rss = np.sum((y_actual - y_predict) ** 2) # Calculate residual sum of squares (RSS)
    r2 = 1 - (rss / tss) # Calculate R-squared
    mse = rss / len(y_actual) # Calculate mean squared error (MSE) as RSS/N
    return EvaluationMetrics(r2, mse, rss, tss)