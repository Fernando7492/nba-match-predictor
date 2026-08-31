import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.transformer import PositionalEncoding, MatchupTransformer
from src.data.sequence_pipeline import NBASequenceDataset
from src.training.sequence_trainer import SequenceModelTrainer

def test_positional_encoding_shape():
    pe = PositionalEncoding(d_model=32, max_len=20)
    x = torch.zeros(4, 10, 32)
    out = pe(x)
    assert out.shape == (4, 10, 32)
    assert not torch.equal(x, out)

def test_matchup_transformer_forward_shapes():
    model = MatchupTransformer(input_dim=15, d_model=32, nhead=4, num_layers=2)
    h_seq = torch.randn(8, 10, 15)
    a_seq = torch.randn(8, 10, 15)

    model.train()
    logits = model(h_seq, a_seq)
    assert logits.shape == (8, 1)

    model.eval()
    p = model.predict_proba(h_seq, a_seq)
    assert p.shape == (8, 1)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0)

def test_matchup_transformer_gradient_flow():
    model = MatchupTransformer(input_dim=10, d_model=16, nhead=2, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    h_seq = torch.randn(4, 5, 10)
    a_seq = torch.randn(4, 5, 10)
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    optimizer.zero_grad()
    logits = model(h_seq, a_seq)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    for p in model.parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))

def test_matchup_transformer_convergence():
    torch.manual_seed(42)
    home = torch.randn(32, 5, 8)
    away = torch.randn(32, 5, 8)
    y = (home[:, -1, 0:1] > away[:, -1, 0:1]).float()

    ds = NBASequenceDataset(home.numpy(), away.numpy(), y.squeeze().numpy())
    loader = DataLoader(ds, batch_size=32)

    model = MatchupTransformer(input_dim=8, d_model=16, nhead=2, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    trainer = SequenceModelTrainer(model, optimizer, patience=60)

    history = trainer.fit(loader, loader, epochs=50)
    assert history["train_loss"][-1] < 0.15
    assert history["train_acc"][-1] >= 0.90
