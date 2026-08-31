import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.sequence_pipeline import NBASequenceDataset

class NBAHybridDataset(Dataset):
    def __init__(
        self,
        tabular_features: np.ndarray,
        home_sequences: torch.Tensor | np.ndarray,
        away_sequences: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray
    ):
        self.tab_x = torch.tensor(tabular_features, dtype=torch.float32)
        self.home_seq = torch.as_tensor(home_sequences, dtype=torch.float32)
        self.away_seq = torch.as_tensor(away_sequences, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        if self.targets.ndim == 1:
            self.targets = self.targets.unsqueeze(1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tab_x[idx], self.home_seq[idx], self.away_seq[idx], self.targets[idx]

def get_hybrid_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    seq_train_ds: NBASequenceDataset,
    seq_val_ds: NBASequenceDataset,
    seq_test_ds: NBASequenceDataset,
    batch_size: int = 64
) -> tuple[DataLoader, DataLoader, DataLoader]:
    x_train_tab = train_df[feature_cols].values
    x_val_tab = val_df[feature_cols].values
    x_test_tab = test_df[feature_cols].values

    train_ds = NBAHybridDataset(x_train_tab, seq_train_ds.home_seq, seq_train_ds.away_seq, seq_train_ds.targets)
    val_ds = NBAHybridDataset(x_val_tab, seq_val_ds.home_seq, seq_val_ds.away_seq, seq_val_ds.targets)
    test_ds = NBAHybridDataset(x_test_tab, seq_test_ds.home_seq, seq_test_ds.away_seq, seq_test_ds.targets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
