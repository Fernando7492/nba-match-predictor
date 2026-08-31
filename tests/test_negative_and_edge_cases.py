import pytest
from pathlib import Path
import numpy as np
import torch
from src.models.mlp import ClassicalMLP
from src.models.deep_mlp import DeepResNetMLP
from src.models.recurrent import DualBranchLSTM
from src.models.transformer import PositionalEncoding, MatchupTransformer
from src.models.ensemble import DeepEnsemblePredictor
from src.utils.config import ProjectPaths
from src.evaluation.visualizer import (
    plot_learning_curves,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_calibration_curves,
    plot_model_comparison
)

def test_predict_proba_preserves_training_state():
    mlp = ClassicalMLP(input_dim=10)
    mlp.train()
    _ = mlp.predict_proba(torch.randn(4, 10))
    assert mlp.training is True

    resnet = DeepResNetMLP(input_dim=10)
    resnet.train()
    _ = resnet.predict_proba(torch.randn(4, 10))
    assert resnet.training is True

    lstm = DualBranchLSTM(input_dim=5)
    lstm.train()
    _ = lstm.predict_proba(torch.randn(4, 6, 5), torch.randn(4, 6, 5))
    assert lstm.training is True

    trans = MatchupTransformer(input_dim=5, d_model=16, nhead=2)
    trans.train()
    _ = trans.predict_proba(torch.randn(4, 6, 5), torch.randn(4, 6, 5))
    assert trans.training is True

def test_ensemble_zero_division_guard():
    ensemble = DeepEnsemblePredictor(weights={"model_a": 1.0})
    mismatched_probas = {"model_x": np.array([0.7, 0.2]), "model_y": np.array([0.6, 0.3])}
    
    preds = ensemble.predict_proba(mismatched_probas)
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))

    empty_preds = ensemble.predict_proba({})
    assert len(empty_preds) == 0

def test_positional_encoding_odd_dimensions():
    pe = PositionalEncoding(d_model=31, max_len=15)
    x = torch.zeros(2, 10, 31)
    out = pe(x)
    assert out.shape == (2, 10, 31)
    assert torch.all(torch.isfinite(out))

def test_project_paths_custom_root_inheritance():
    custom_root = Path("/tmp/custom_nba_project")
    paths = ProjectPaths(root=custom_root)
    assert paths.data_raw == custom_root / "data" / "raw"
    assert paths.data_processed == custom_root / "data" / "processed"
    assert paths.outputs_models == custom_root / "outputs" / "models"
    assert paths.outputs_figures == custom_root / "outputs" / "figures"

def test_visualizer_empty_probas_safe(tmp_path: Path):
    y_true = np.array([1, 0, 1])
    plot_confusion_matrices(y_true, {}, tmp_path / "cm.png")
    plot_roc_curves(y_true, {}, tmp_path / "roc.png")
    plot_calibration_curves(y_true, {}, tmp_path / "calib.png")
    plot_learning_curves({}, tmp_path / "learn.png")
