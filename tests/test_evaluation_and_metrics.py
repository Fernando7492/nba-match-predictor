import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.evaluation.metrics import compute_all_metrics, compute_expected_calibration_error
from src.evaluation.visualizer import (
    plot_learning_curves,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_calibration_curves,
    plot_model_comparison
)

def test_metrics_values_and_bounds():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.6, 0.4])

    metrics = compute_all_metrics(y_true, y_prob)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["log_loss"] > 0.0
    assert 0.0 <= metrics["ece"] <= 1.0

def test_ece_perfect_calibration():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])
    ece = compute_expected_calibration_error(y_true, y_prob)
    assert ece == pytest.approx(0.0)

def test_visualizer_plots_generation(tmp_path: Path):
    histories = {
        "Model A": {
            "train_loss": [0.7, 0.5, 0.3],
            "val_loss": [0.75, 0.55, 0.35],
            "train_acc": [0.6, 0.7, 0.8],
            "val_acc": [0.58, 0.68, 0.78]
        }
    }
    y_true = np.array([1, 0, 1, 0, 1, 0])
    model_probas = {
        "Model A": np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1])
    }
    results_df = pd.DataFrame([
        {"model": "Model A", "accuracy": 0.8, "f1_score": 0.79, "roc_auc": 0.85, "brier_score": 0.15}
    ])

    lc_path = tmp_path / "lc.png"
    cm_path = tmp_path / "cm.png"
    roc_path = tmp_path / "roc.png"
    cal_path = tmp_path / "cal.png"
    bar_path = tmp_path / "bar.png"

    plot_learning_curves(histories, lc_path)
    plot_confusion_matrices(y_true, model_probas, cm_path)
    plot_roc_curves(y_true, model_probas, roc_path)
    plot_calibration_curves(y_true, model_probas, cal_path)
    plot_model_comparison(results_df, bar_path)

    assert lc_path.exists() and lc_path.stat().st_size > 1000
    assert cm_path.exists() and cm_path.stat().st_size > 1000
    assert roc_path.exists() and roc_path.stat().st_size > 1000
    assert cal_path.exists() and cal_path.stat().st_size > 1000
    assert bar_path.exists() and bar_path.stat().st_size > 1000
