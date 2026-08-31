import torch
import torch.nn as nn

class ClassicalMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 1)
        )
        self.net = self.network
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return torch.sigmoid(self.forward(x))
        finally:
            if was_training:
                self.train()
