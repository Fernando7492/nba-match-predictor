import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class MatchupTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 21,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        rep_dim = d_model * 2
        fused_dim = rep_dim * 4

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def _encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.proj(seq)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        avg_pool = out.mean(dim=1)
        last_step = out[:, -1, :]
        return torch.cat([avg_pool, last_step], dim=1)

    def forward(self, home_seq: torch.Tensor, away_seq: torch.Tensor) -> torch.Tensor:
        rep_h = self._encode_sequence(home_seq)
        rep_a = self._encode_sequence(away_seq)

        diff = rep_h - rep_a
        mult = rep_h * rep_a

        fused = torch.cat([rep_h, rep_a, diff, mult], dim=1)
        return self.head(fused)

    def predict_proba(self, home_seq: torch.Tensor, away_seq: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(home_seq, away_seq)
            return torch.sigmoid(logits)
