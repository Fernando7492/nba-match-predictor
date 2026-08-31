import torch
import torch.nn as nn

class DualBranchLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 21,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        fusion_dim = (hidden_dim * 2) * 4
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, home_seq: torch.Tensor, away_seq: torch.Tensor) -> torch.Tensor:
        out_h, _ = self.lstm(home_seq)
        out_a, _ = self.lstm(away_seq)

        rep_h = out_h[:, -1, :]
        rep_a = out_a[:, -1, :]

        diff = rep_h - rep_a
        mult = rep_h * rep_a

        fused = torch.cat([rep_h, rep_a, diff, mult], dim=1)
        return self.head(fused)

    def predict_proba(self, home_seq: torch.Tensor, away_seq: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(home_seq, away_seq)
            return torch.sigmoid(logits)
