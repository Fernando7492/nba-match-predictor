import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class OptimalThresholdTuner:
    def __init__(self, metric: str = "f1", min_thresh: float = 0.40, max_thresh: float = 0.60, steps: int = 101):
        self.metric = metric
        self.thresholds = np.linspace(min_thresh, max_thresh, steps)
        self.best_threshold: float = 0.50

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "OptimalThresholdTuner":
        y_true_arr = np.asarray(y_true).ravel()
        y_prob_arr = np.asarray(y_prob).ravel()

        best_score = -float("inf")
        best_t = 0.50

        for t in self.thresholds:
            preds = (y_prob_arr >= t).astype(int)
            if self.metric == "f1":
                score = f1_score(y_true_arr, preds, zero_division=0)
            elif self.metric == "accuracy":
                score = accuracy_score(y_true_arr, preds)
            else:
                score = 0.5 * (f1_score(y_true_arr, preds, zero_division=0) + accuracy_score(y_true_arr, preds))

            if score > best_score:
                best_score = score
                best_t = float(t)

        self.best_threshold = best_t
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        return (np.asarray(y_prob).ravel() >= self.best_threshold).astype(int)
