import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.baseline import HomeCourtBaseline, LogisticRegressionBaseline, RandomForestBaseline
from src.models.mlp import ClassicalMLP
from src.training.trainer import ModelTrainer

def test_baselines_execution():
    np.random.seed(42)
    x = np.random.randn(100, 10)
    y = np.random.binomial(1, 0.6, size=(100, 1))

    home_baseline = HomeCourtBaseline().fit(x, y)
    p_home = home_baseline.predict_proba(x)
    assert p_home.shape == (100, 2)
    assert np.all(p_home >= 0.0) and np.all(p_home <= 1.0)
    assert home_baseline.predict(x).shape == (100,)

    lr_baseline = LogisticRegressionBaseline().fit(x, y)
    p_lr = lr_baseline.predict_proba(x)
    assert p_lr.shape == (100, 2)
    assert lr_baseline.predict(x).shape == (100,)

    rf_baseline = RandomForestBaseline().fit(x, y)
    p_rf = rf_baseline.predict_proba(x)
    assert p_rf.shape == (100, 2)
    assert rf_baseline.predict(x).shape == (100,)

def test_classical_mlp_forward_shapes():
    model = ClassicalMLP(input_dim=20, hidden_dim_1=32, hidden_dim_2=16)
    x = torch.randn(8, 20)
    logits = model(x)
    assert logits.shape == (8, 1)
    proba = model.predict_proba(x)
    assert proba.shape == (8, 1)
    assert torch.all(proba >= 0.0) and torch.all(proba <= 1.0)

def test_classical_mlp_gradient_flow():
    model = ClassicalMLP(input_dim=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randn(16, 15)
    y = torch.randint(0, 2, (16, 1)).float()

    init_param = model.network[0].weight.clone()

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    updated_param = model.network[0].weight
    assert not torch.equal(init_param, updated_param)

def test_trainer_mini_batch_convergence():
    torch.manual_seed(42)
    x = torch.randn(32, 10)
    y = (x[:, 0:1] > 0.0).float()
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=32)

    model = ClassicalMLP(input_dim=10, hidden_dim_1=32, hidden_dim_2=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    trainer = ModelTrainer(model, optimizer, patience=100)

    history = trainer.fit(loader, loader, epochs=60)
    assert history["train_loss"][-1] < 0.08
    assert history["train_acc"][-1] >= 0.95
