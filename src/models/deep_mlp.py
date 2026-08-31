import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int | None = None, dim: int | None = None, dropout: float = 0.2):
        super().__init__()
        h_dim = hidden_dim or dim or 128
        self.block = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim)
        )
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

class DeepResNetMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return torch.sigmoid(self.forward(x))
        finally:
            if was_training:
                self.train()
