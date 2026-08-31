import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

class DeepEnsemblePredictor:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {}

    def fit_weights(self, val_probas: dict[str, np.ndarray], y_val: np.ndarray) -> "DeepEnsemblePredictor":
        model_names = list(val_probas.keys())
        n_models = len(model_names)
        
        prob_matrix = np.column_stack([val_probas[name].ravel() for name in model_names])
        y_val_arr = np.asarray(y_val).ravel()

        def loss_fn(w: np.ndarray) -> float:
            p_ens = prob_matrix @ w
            return float(brier_score_loss(y_val_arr, p_ens))

        init_w = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(loss_fn, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        
        if res.success:
            opt_w = res.x
        else:
            opt_w = init_w

        self.weights = {name: float(w) for name, w in zip(model_names, opt_w)}
        return self

    def predict_proba(self, model_probas: dict[str, np.ndarray]) -> np.ndarray:
        if not self.weights:
            n_models = len(model_probas)
            weights = {name: 1.0 / n_models for name in model_probas}
        else:
            weights = self.weights

        weighted_prob = np.zeros(len(next(iter(model_probas.values()))))
        total_w = sum(weights.get(name, 0.0) for name in model_probas)

        for name, prob in model_probas.items():
            w = weights.get(name, 0.0) / total_w
            weighted_prob += w * np.asarray(prob).ravel()

        return np.clip(weighted_prob, 0.0, 1.0)

    def predict(self, model_probas: dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(model_probas) >= threshold).astype(int)
