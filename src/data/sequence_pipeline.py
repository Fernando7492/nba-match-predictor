from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from src.utils.config import ProjectPaths

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
        self.stat_cols = [
            "PTS", "PTS_ALLOWED", "FGM", "FGA", "FG_PCT",
            "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
            "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
            "PLUS_MINUS", "WIN"
        ]

    def build_sequences(
        self,
        raw_df: pd.DataFrame,
        train_seasons: list[str] | None = None,
        val_seasons: list[str] | None = None,
        test_seasons: list[str] | None = None
    ) -> tuple[NBASequenceDataset, NBASequenceDataset, NBASequenceDataset]:
        train_seasons = train_seasons or ["2018-19", "2019-20", "2020-21", "2021-22"]
        val_seasons = val_seasons or ["2022-23"]
        test_seasons = test_seasons or ["2023-24"]

        df = raw_df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ").astype(int)
        df["WIN"] = (df["WL"] == "W").astype(float)
        
        for pct_col in ["FG_PCT", "FG3_PCT", "FT_PCT"]:
            if pct_col in df.columns:
                df[pct_col] = df[pct_col].fillna(0.0)

        df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

        opponents = df[["GAME_ID", "TEAM_ID", "PTS"]].rename(
            columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "PTS_ALLOWED"}
        )
        merged = df.merge(opponents, on="GAME_ID")
        merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()
        merged = merged.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

        train_stat_rows = merged[merged["SEASON"].isin(train_seasons)][self.stat_cols].values
        self.scaler.fit(train_stat_rows)

        scaled_stats = self.scaler.transform(merged[self.stat_cols].values)
        for i, col in enumerate(self.stat_cols):
            merged[f"SCALED_{col}"] = scaled_stats[:, i]

        scaled_cols = [f"SCALED_{c}" for c in self.stat_cols]

        team_history: dict[int, list[dict]] = {}
        for team_id in merged["TEAM_ID"].unique():
            team_history[team_id] = []

        all_games = (
            merged[merged["IS_HOME"] == 1][["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "WIN"]]
            .rename(columns={"TEAM_ID": "HOME_TEAM_ID", "WIN": "TARGET_HOME_W"})
            .merge(
                merged[merged["IS_HOME"] == 0][["GAME_ID", "TEAM_ID"]].rename(
                    columns={"TEAM_ID": "AWAY_TEAM_ID"}
                ),
                on="GAME_ID"
            )
            .sort_values("GAME_DATE")
            .reset_index(drop=True)
        )

        team_game_stats = {}
        for _, row in merged.iterrows():
            g_id = row["GAME_ID"]
            t_id = row["TEAM_ID"]
            stats_vec = row[scaled_cols].values.astype(np.float32)
            team_game_stats[(g_id, t_id)] = stats_vec

        split_data: dict[str, dict[str, list]] = {
            "train": {"home": [], "away": [], "target": []},
            "val": {"home": [], "away": [], "target": []},
            "test": {"home": [], "away": [], "target": []}
        }

        d = len(scaled_cols)
        pad_vec = np.zeros(d, dtype=np.float32)

        for _, match in all_games.iterrows():
            g_id = match["GAME_ID"]
            h_id = match["HOME_TEAM_ID"]
            a_id = match["AWAY_TEAM_ID"]
            season = match["SEASON"]
            target = match["TARGET_HOME_W"]

            h_hist = team_history[h_id]
            a_hist = team_history[a_id]

            if len(h_hist) >= self.sequence_length:
                h_seq = np.array(h_hist[-self.sequence_length:], dtype=np.float32)
            else:
                pad_len = self.sequence_length - len(h_hist)
                h_seq = np.array([pad_vec] * pad_len + h_hist, dtype=np.float32)

            if len(a_hist) >= self.sequence_length:
                a_seq = np.array(a_hist[-self.sequence_length:], dtype=np.float32)
            else:
                pad_len = self.sequence_length - len(a_hist)
                a_seq = np.array([pad_vec] * pad_len + a_hist, dtype=np.float32)

            split_name = "train" if season in train_seasons else ("val" if season in val_seasons else "test")
            split_data[split_name]["home"].append(h_seq)
            split_data[split_name]["away"].append(a_seq)
            split_data[split_name]["target"].append(target)

            team_history[h_id].append(team_game_stats[(g_id, h_id)])
            team_history[a_id].append(team_game_stats[(g_id, a_id)])

        train_ds = NBASequenceDataset(
            np.array(split_data["train"]["home"]),
            np.array(split_data["train"]["away"]),
            np.array(split_data["train"]["target"])
        )
        val_ds = NBASequenceDataset(
            np.array(split_data["val"]["home"]),
            np.array(split_data["val"]["away"]),
            np.array(split_data["val"]["target"])
        )
        test_ds = NBASequenceDataset(
            np.array(split_data["test"]["home"]),
            np.array(split_data["test"]["away"]),
            np.array(split_data["test"]["target"])
        )

        return train_ds, val_ds, test_ds
