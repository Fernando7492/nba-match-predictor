import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class NBATabularDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

def get_tabular_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int = 64,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    x_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["TARGET_HOME_W"].values.astype(np.float32)

    x_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["TARGET_HOME_W"].values.astype(np.float32)

    x_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["TARGET_HOME_W"].values.astype(np.float32)

    train_ds = NBATabularDataset(x_train, y_train)
    val_ds = NBATabularDataset(x_val, y_val)
    test_ds = NBATabularDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
