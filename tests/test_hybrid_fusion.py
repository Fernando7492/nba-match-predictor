import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.hybrid_fusion import CrossAttentionFusionNet
from src.data.hybrid_dataset import NBAHybridDataset
from src.training.hybrid_trainer import HybridModelTrainer

def test_cross_attention_fusion_forward_shapes():
    model = CrossAttentionFusionNet(tabular_dim=50, sequence_stat_dim=15, d_model=32, nhead=4)
    tab_x = torch.randn(8, 50)
    home_seq = torch.randn(8, 10, 15)
    away_seq = torch.randn(8, 10, 15)

    model.train()
    logits = model(tab_x, home_seq, away_seq)
    assert logits.shape == (8, 1)

    model.eval()
    p = model.predict_proba(tab_x, home_seq, away_seq)
    assert p.shape == (8, 1)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0)

def test_cross_attention_fusion_gradient_flow():
    model = CrossAttentionFusionNet(tabular_dim=20, sequence_stat_dim=8, d_model=16, nhead=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    tab_x = torch.randn(4, 20)
    h_seq = torch.randn(4, 5, 8)
    a_seq = torch.randn(4, 5, 8)
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    optimizer.zero_grad()
    logits = model(tab_x, h_seq, a_seq)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    for p in model.parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))

def test_cross_attention_fusion_state_preservation():
    model = CrossAttentionFusionNet(tabular_dim=10, sequence_stat_dim=5, d_model=16, nhead=2)
    model.train()
    _ = model.predict_proba(torch.randn(2, 10), torch.randn(2, 5, 5), torch.randn(2, 5, 5))
    assert model.training is True

def test_hybrid_trainer_convergence_on_synthetic_data():
    torch.manual_seed(42)
    tab_x = torch.randn(32, 12)
    h_seq = torch.randn(32, 5, 6)
    a_seq = torch.randn(32, 5, 6)
    y = (tab_x[:, 0:1] > 0.0).float()

    ds = NBAHybridDataset(tab_x.numpy(), h_seq.numpy(), a_seq.numpy(), y.squeeze().numpy())
    loader = DataLoader(ds, batch_size=32)

    model = CrossAttentionFusionNet(tabular_dim=12, sequence_stat_dim=6, d_model=16, nhead=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    trainer = HybridModelTrainer(model, opt, patience=60)

    hist = trainer.fit(loader, loader, epochs=60)
    assert hist["train_loss"][-1] < 0.25
    assert hist["train_acc"][-1] >= 0.85
