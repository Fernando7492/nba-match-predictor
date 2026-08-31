from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from src.utils.config import (
    ProjectPaths,
    DEFAULT_SEASONS_TRAIN,
    DEFAULT_SEASONS_VAL,
    DEFAULT_SEASONS_TEST,
    BASE_STAT_COLS
)
from src.data.common import prepare_raw_team_logs

class NBASequenceDataset(Dataset):
    def __init__(
        self,
        home_sequences: np.ndarray,
        away_sequences: np.ndarray,
        targets: np.ndarray
    ):
        self.home_seq = torch.tensor(home_sequences, dtype=torch.float32)
        self.away_seq = torch.tensor(away_sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.home_seq[idx], self.away_seq[idx], self.targets[idx]

class NBASequencePipeline:
    def __init__(
        self,
        paths: ProjectPaths | None = None,
        sequence_length: int = 10
    ):
        self.paths = paths or ProjectPaths()
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.stat_cols = list(BASE_STAT_COLS)

    def build_sequences(
        self,
        raw_df: pd.DataFrame,
        valid_game_ids: set[str] | None = None,
        train_seasons: list[str] | None = None,
        val_seasons: list[str] | None = None,
        test_seasons: list[str] | None = None
    ) -> tuple[NBASequenceDataset, NBASequenceDataset, NBASequenceDataset]:
        train_seasons = train_seasons or DEFAULT_SEASONS_TRAIN
        val_seasons = val_seasons or DEFAULT_SEASONS_VAL
        test_seasons = test_seasons or DEFAULT_SEASONS_TEST

        merged = prepare_raw_team_logs(raw_df)

        train_stat_rows = merged[merged["SEASON"].isin(train_seasons)][self.stat_cols].values
        self.scaler.fit(train_stat_rows)

        scaled_matrix = self.scaler.transform(merged[self.stat_cols].values).astype(np.float32)
        
        team_ids = merged["TEAM_ID"].values
        game_ids = merged["GAME_ID"].astype(str).values
        
        team_game_stats = {
            (game_ids[i], team_ids[i]): scaled_matrix[i]
            for i in range(len(merged))
        }

        all_games = (
            merged[merged["IS_HOME"] == 1][["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "WIN"]]
            .rename(columns={"TEAM_ID": "HOME_TEAM_ID", "WIN": "TARGET_HOME_W"})
            .merge(
                merged[merged["IS_HOME"] == 0][["GAME_ID", "TEAM_ID"]].rename(
                    columns={"TEAM_ID": "AWAY_TEAM_ID"}
                ),
                on="GAME_ID"
            )
            .sort_values(["GAME_DATE", "GAME_ID"])
            .reset_index(drop=True)
        )

        team_history: dict[int, list[np.ndarray]] = {
            t_id: [] for t_id in np.unique(team_ids)
        }

        split_data: dict[str, dict[str, list[np.ndarray]]] = {
            "train": {"home": [], "away": [], "target": []},
            "val": {"home": [], "away": [], "target": []},
            "test": {"home": [], "away": [], "target": []}
        }

        d = len(self.stat_cols)
        pad_vec = np.zeros(d, dtype=np.float32)

        g_ids = all_games["GAME_ID"].astype(str).values
        h_ids = all_games["HOME_TEAM_ID"].values
        a_ids = all_games["AWAY_TEAM_ID"].values
        seasons = all_games["SEASON"].values
        targets = all_games["TARGET_HOME_W"].values.astype(np.float32)

        train_set = set(train_seasons)
        val_set = set(val_seasons)

        for i in range(len(all_games)):
            g_id = g_ids[i]
            h_id = h_ids[i]
            a_id = a_ids[i]
            season = seasons[i]
            target = targets[i]

            h_hist = team_history[h_id]
            a_hist = team_history[a_id]

            if len(h_hist) >= self.sequence_length:
                h_seq = np.stack(h_hist[-self.sequence_length:], axis=0)
            else:
                pad_len = self.sequence_length - len(h_hist)
                padded_list = [pad_vec] * pad_len + h_hist
                h_seq = np.stack(padded_list, axis=0) if padded_list else np.zeros((self.sequence_length, d), dtype=np.float32)

            if len(a_hist) >= self.sequence_length:
                a_seq = np.stack(a_hist[-self.sequence_length:], axis=0)
            else:
                pad_len = self.sequence_length - len(a_hist)
                padded_list = [pad_vec] * pad_len + a_hist
                a_seq = np.stack(padded_list, axis=0) if padded_list else np.zeros((self.sequence_length, d), dtype=np.float32)

            if valid_game_ids is None or g_id in valid_game_ids:
                split_name = "train" if season in train_set else ("val" if season in val_set else "test")
                split_data[split_name]["home"].append(h_seq)
                split_data[split_name]["away"].append(a_seq)
                split_data[split_name]["target"].append(target)

            team_history[h_id].append(team_game_stats[(g_id, h_id)])
            team_history[a_id].append(team_game_stats[(g_id, a_id)])

        train_ds = NBASequenceDataset(
            np.array(split_data["train"]["home"], dtype=np.float32),
            np.array(split_data["train"]["away"], dtype=np.float32),
            np.array(split_data["train"]["target"], dtype=np.float32)
        )
        val_ds = NBASequenceDataset(
            np.array(split_data["val"]["home"], dtype=np.float32),
            np.array(split_data["val"]["away"], dtype=np.float32),
            np.array(split_data["val"]["target"], dtype=np.float32)
        )
        test_ds = NBASequenceDataset(
            np.array(split_data["test"]["home"], dtype=np.float32),
            np.array(split_data["test"]["away"], dtype=np.float32),
            np.array(split_data["test"]["target"], dtype=np.float32)
        )

        return train_ds, val_ds, test_ds
