import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, f1_score

class DeepEnsemblePredictor:
    def __init__(self, weights: dict[str, float] | None = None, optimal_threshold: float = 0.5):
        self.weights = weights or {}
        self.optimal_threshold = optimal_threshold

    def fit_weights(self, val_probas: dict[str, np.ndarray], y_val: np.ndarray) -> "DeepEnsemblePredictor":
        model_names = list(val_probas.keys())
        n_models = len(model_names)
        
        prob_matrix = np.column_stack([np.clip(val_probas[name].ravel(), 0.0, 1.0) for name in model_names])
        y_val_arr = np.asarray(y_val).ravel()

        def loss_fn(w: np.ndarray) -> float:
            p_ens = np.clip(prob_matrix @ w, 0.0, 1.0)
            return float(brier_score_loss(y_val_arr, p_ens))

        init_w = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(loss_fn, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        
        if res.success and np.all(np.isfinite(res.x)):
            opt_w = np.clip(res.x, 0.0, 1.0)
            s = np.sum(opt_w)
            opt_w = opt_w / s if s > 0 else init_w
        else:
            opt_w = init_w

        self.weights = {name: float(w) for name, w in zip(model_names, opt_w)}
        
        p_val_ens = self.predict_proba(val_probas)
        best_t = 0.50
        best_f1 = -1.0
        for t in np.linspace(0.40, 0.60, 41):
            f1 = f1_score(y_val_arr, (p_val_ens >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        self.optimal_threshold = best_t

        return self

    def predict_proba(self, model_probas: dict[str, np.ndarray]) -> np.ndarray:
        if not model_probas:
            return np.array([], dtype=float)

        first_arr = np.asarray(next(iter(model_probas.values())))
        n_samples = len(first_arr)
        
        weights = self.weights if self.weights else {name: 1.0 / len(model_probas) for name in model_probas}
        total_w = sum(weights.get(name, 0.0) for name in model_probas)

        if total_w <= 0.0:
            weights = {name: 1.0 / len(model_probas) for name in model_probas}
            total_w = 1.0

        weighted_prob = np.zeros(n_samples, dtype=float)
        for name, prob in model_probas.items():
            w = weights.get(name, 0.0) / total_w
            weighted_prob += w * np.clip(np.asarray(prob).ravel(), 0.0, 1.0)

        return np.clip(weighted_prob, 0.0, 1.0)

    def predict(self, model_probas: dict[str, np.ndarray], threshold: float | None = None) -> np.ndarray:
        t = threshold if threshold is not None else self.optimal_threshold
        return (self.predict_proba(model_probas) >= t).astype(int)
