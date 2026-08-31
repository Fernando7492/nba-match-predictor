import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term[:pe[:, 0::2].size(1)])
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class MatchupTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
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
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode_branch(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.proj(seq) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer(x)
        mean_pool = encoded.mean(dim=1)
        last_state = encoded[:, -1, :]
        return torch.cat([mean_pool, last_state], dim=-1)

    def forward(self, home_seq: torch.Tensor, away_seq: torch.Tensor) -> torch.Tensor:
        rep_h = self._encode_branch(home_seq)
        rep_a = self._encode_branch(away_seq)

        diff = rep_h - rep_a
        mult = rep_h * rep_a
        fused = torch.cat([rep_h, rep_a, diff, mult], dim=-1)

        return self.head(fused)

    def predict_proba(self, home_seq: torch.Tensor, away_seq: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return torch.sigmoid(self.forward(home_seq, away_seq))
        finally:
            if was_training:
                self.train()
