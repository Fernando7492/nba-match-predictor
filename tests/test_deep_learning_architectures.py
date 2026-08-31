import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.deep_mlp import DeepResNetMLP, ResidualBlock
from src.models.recurrent import DualBranchLSTM
from src.training.trainer import ModelTrainer
from src.training.sequence_trainer import SequenceModelTrainer
from src.data.sequence_pipeline import NBASequenceDataset, NBASequencePipeline
from src.data.collector import NBADataCollector

def test_residual_block_forward():
    block = ResidualBlock(dim=32, dropout=0.1)
    x = torch.randn(8, 32)
    out = block(x)
    assert out.shape == (8, 32)

def test_deep_resnet_mlp_forward_and_eval():
    model = DeepResNetMLP(input_dim=50, hidden_dim=64, num_blocks=3, dropout=0.2)
    x = torch.randn(16, 50)
    
    model.train()
    out_train = model(x)
    assert out_train.shape == (16, 1)

    model.eval()
    out_eval = model(x)
    assert out_eval.shape == (16, 1)

    p = model.predict_proba(x)
    assert p.shape == (16, 1)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0)

def test_deep_resnet_gradient_norm_stability():
    model = DeepResNetMLP(input_dim=30, hidden_dim=64, num_blocks=2)
    x = torch.randn(16, 30)
    y = torch.randint(0, 2, (16, 1)).float()
    criterion = nn.BCEWithLogitsLoss()

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            assert not np.isnan(norm)
            assert norm < 50.0

def test_dual_branch_lstm_forward():
    model = DualBranchLSTM(input_dim=15, hidden_dim=32, num_layers=2)
    home_seq = torch.randn(8, 10, 15)
    away_seq = torch.randn(8, 10, 15)

    logits = model(home_seq, away_seq)
    assert logits.shape == (8, 1)

    p = model.predict_proba(home_seq, away_seq)
    assert p.shape == (8, 1)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0)

def test_sequence_trainer_convergence():
    torch.manual_seed(42)
    home = torch.randn(32, 5, 8)
    away = torch.randn(32, 5, 8)
    y = (home[:, -1, 0:1] > away[:, -1, 0:1]).float()

    ds = NBASequenceDataset(home.numpy(), away.numpy(), y.squeeze().numpy())
    loader = DataLoader(ds, batch_size=32)

    model = DualBranchLSTM(input_dim=8, hidden_dim=16, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    trainer = SequenceModelTrainer(model, optimizer, patience=60)

    history = trainer.fit(loader, loader, epochs=50)
    assert history["train_loss"][-1] < 0.15
    assert history["train_acc"][-1] >= 0.90

def test_sequence_pipeline_real_data():
    collector = NBADataCollector()
    raw_df = collector.collect_all_seasons()
    pipeline = NBASequencePipeline(sequence_length=5)
    train_ds, val_ds, test_ds = pipeline.build_sequences(raw_df)

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    h, a, y = train_ds[0]
    assert h.shape == (5, len(pipeline.stat_cols))
    assert a.shape == (5, len(pipeline.stat_cols))
    assert y.shape == (1,)
