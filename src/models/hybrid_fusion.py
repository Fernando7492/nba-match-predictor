import math
import torch
import torch.nn as nn
from src.models.transformer import PositionalEncoding
from src.models.deep_mlp import ResidualBlock

class CrossAttentionFusionNet(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        sequence_stat_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.d_model = d_model

        self.tab_proj = nn.Sequential(
            nn.Linear(tabular_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.Mish(),
            nn.Dropout(dropout)
        )
        self.tab_res = ResidualBlock(hidden_dim=d_model, dropout=dropout)

        self.seq_proj = nn.Linear(sequence_stat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        self.cross_attn_home = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_away = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        fused_dim = d_model * 5
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        tab_x: torch.Tensor,
        home_seq: torch.Tensor,
        away_seq: torch.Tensor
    ) -> torch.Tensor:
        tab_h = self.tab_proj(tab_x)
        tab_feat = self.tab_res(tab_h)
        query = tab_feat.unsqueeze(1)

        h_seq = self.seq_proj(home_seq) * math.sqrt(self.d_model)
        h_seq = self.pos_encoder(h_seq)
        
        a_seq = self.seq_proj(away_seq) * math.sqrt(self.d_model)
        a_seq = self.pos_encoder(a_seq)

        attn_h, _ = self.cross_attn_home(query, h_seq, h_seq)
        attn_a, _ = self.cross_attn_away(query, a_seq, a_seq)

        ctx_h = attn_h.squeeze(1)
        ctx_a = attn_a.squeeze(1)

        diff = ctx_h - ctx_a
        mult = ctx_h * ctx_a
        fused = torch.cat([tab_feat, ctx_h, ctx_a, diff, mult], dim=-1)

        return self.head(fused)

    def predict_proba(
        self,
        tab_x: torch.Tensor,
        home_seq: torch.Tensor,
        away_seq: torch.Tensor
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return torch.sigmoid(self.forward(tab_x, home_seq, away_seq))
        finally:
            if was_training:
                self.train()
