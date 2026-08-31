import torch
import torch.nn as nn

class DualBranchLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
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
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1.0)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, home_seq: torch.Tensor, away_seq: torch.Tensor) -> torch.Tensor:
        out_h, _ = self.lstm(home_seq)
        out_a, _ = self.lstm(away_seq)

        rep_h = torch.cat([out_h[:, -1, :self.hidden_dim], out_h[:, 0, self.hidden_dim:]], dim=-1)
        rep_a = torch.cat([out_a[:, -1, :self.hidden_dim], out_a[:, 0, self.hidden_dim:]], dim=-1)

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
