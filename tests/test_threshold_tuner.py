import pytest
import numpy as np
from src.evaluation.threshold_tuner import OptimalThresholdTuner
from src.models.ensemble import DeepEnsemblePredictor

def test_threshold_tuner_optimization():
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.55, 0.60, 0.52, 0.48, 0.40, 0.30, 0.58, 0.45, 0.70, 0.35])

    tuner = OptimalThresholdTuner(metric="f1").fit(y_true, y_prob)
    preds = tuner.predict(y_prob)

    assert 0.40 <= tuner.best_threshold <= 0.60
    assert len(preds) == len(y_true)
    assert set(preds).issubset({0, 1})

def test_ensemble_with_optimal_threshold():
    y_val = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    p1 = np.array([0.85, 0.25, 0.80, 0.75, 0.30, 0.15, 0.85, 0.20, 0.90, 0.10])
    p2 = np.array([0.65, 0.45, 0.60, 0.55, 0.40, 0.45, 0.65, 0.40, 0.60, 0.40])

    val_probas = {"m1": p1, "m2": p2}
    ensemble = DeepEnsemblePredictor().fit_weights(val_probas, y_val)

    assert 0.40 <= ensemble.optimal_threshold <= 0.60
    preds = ensemble.predict(val_probas)
    assert len(preds) == len(y_val)
