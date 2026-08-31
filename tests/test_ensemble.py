import pytest
import numpy as np
from src.models.ensemble import DeepEnsemblePredictor

def test_ensemble_weights_optimization():
    np.random.seed(42)
    y_val = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    
    p1 = np.array([0.9, 0.2, 0.8, 0.7, 0.3, 0.1, 0.85, 0.2, 0.9, 0.1])
    p2 = np.array([0.6, 0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4, 0.6, 0.4])

    val_probas = {"model_good": p1, "model_weak": p2}
    ensemble = DeepEnsemblePredictor().fit_weights(val_probas, y_val)

    assert len(ensemble.weights) == 2
    assert pytest.approx(sum(ensemble.weights.values())) == 1.0
    assert ensemble.weights["model_good"] > ensemble.weights["model_weak"]

def test_ensemble_prediction_bounds():
    p1 = np.array([0.8, 0.2, 0.9])
    p2 = np.array([0.7, 0.3, 0.85])
    test_probas = {"m1": p1, "m2": p2}

    ensemble = DeepEnsemblePredictor(weights={"m1": 0.6, "m2": 0.4})
    p_ens = ensemble.predict_proba(test_probas)
    preds = ensemble.predict(test_probas)

    assert p_ens.shape == (3,)
    assert np.all(p_ens >= 0.0) and np.all(p_ens <= 1.0)
    assert set(preds).issubset({0, 1})
