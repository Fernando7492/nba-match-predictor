import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.training.losses import LabelSmoothingBCELoss
from src.training.trainer import ModelTrainer
from src.models.mlp import ClassicalMLP

def test_label_smoothing_loss_bounds():
    criterion = LabelSmoothingBCELoss(smoothing=0.1)
    logits = torch.tensor([[2.0], [-2.0]], requires_grad=True)
    targets = torch.tensor([[1.0], [0.0]])

    loss = criterion(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

def test_label_smoothing_loss_gradient_flow():
    criterion = LabelSmoothingBCELoss(smoothing=0.08)
    logits = torch.randn(8, 1, requires_grad=True)
    targets = torch.randint(0, 2, (8, 1)).float()

    loss = criterion(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))

def test_cosine_annealing_with_trainer():
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32, 1)).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    model = ClassicalMLP(input_dim=10, hidden_dim_1=16, hidden_dim_2=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    criterion = LabelSmoothingBCELoss(smoothing=0.08)

    trainer = ModelTrainer(model, optimizer, criterion=criterion, scheduler=scheduler, patience=10)
    hist = trainer.fit(loader, loader, epochs=10)
    assert len(hist["train_loss"]) == 10
