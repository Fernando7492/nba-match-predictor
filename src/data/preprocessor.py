import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from src.utils.config import ProjectPaths

class NBAPreprocessor:
    def __init__(
        self,
        paths: ProjectPaths | None = None,
        rolling_windows: list[int] | None = None
    ):
        self.paths = paths or ProjectPaths()
        self.rolling_windows = rolling_windows or [3, 7, 14]
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    def _prepare_team_logs(self, raw_df: pd.DataFrame) -> pd.DataFrame:
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
        return merged

    def _compute_rolling_features(self, team_df: pd.DataFrame) -> pd.DataFrame:
        base_stats = [
            "PTS", "PTS_ALLOWED", "FGM", "FGA", "FG_PCT",
            "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
            "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
            "PLUS_MINUS", "WIN"
        ]

        grouped = team_df.groupby("TEAM_ID")
        team_df["REST_DAYS"] = grouped["GAME_DATE"].diff().dt.total_seconds() / 86400.0
        team_df["REST_DAYS"] = team_df["REST_DAYS"].fillna(7.0).clip(lower=0.0, upper=10.0)
        team_df["BACK_TO_BACK"] = (team_df["REST_DAYS"] <= 1.0).astype(float)

        for col in base_stats:
            shifted = grouped[col].shift(1)
            for w in self.rolling_windows:
                roll = shifted.groupby(team_df["TEAM_ID"]).rolling(window=w, min_periods=1).mean()
                team_df[f"{col}_ROLL_{w}"] = roll.reset_index(level=0, drop=True)

        return team_df

    def _build_matchup_dataset(self, featured_team_df: pd.DataFrame) -> pd.DataFrame:
        home = featured_team_df[featured_team_df["IS_HOME"] == 1].copy()
        away = featured_team_df[featured_team_df["IS_HOME"] == 0].copy()

        suffixes = ("_HOME", "_AWAY")
        matchups = home.merge(
            away,
            on="GAME_ID",
            suffixes=suffixes
        )

        matchups["TARGET_HOME_W"] = matchups["WIN_HOME"].astype(int)
        matchups["GAME_DATE"] = matchups["GAME_DATE_HOME"]
        matchups["SEASON"] = matchups["SEASON_HOME"]

        rolling_cols = [
            col for col in home.columns if "_ROLL_" in col or col in ["REST_DAYS", "BACK_TO_BACK"]
        ]

        diff_cols = []
        for col in rolling_cols:
            home_col = f"{col}_HOME"
            away_col = f"{col}_AWAY"
            diff_col = f"{col}_DIFF"
            matchups[diff_col] = matchups[home_col] - matchups[away_col]
            diff_cols.append(diff_col)

        feature_cols = (
            [f"{c}_HOME" for c in rolling_cols]
            + [f"{c}_AWAY" for c in rolling_cols]
            + diff_cols
        )

        matchups = matchups.dropna(subset=feature_cols).reset_index(drop=True)
        self.feature_columns = feature_cols
        return matchups

    def process_and_split(
        self,
        raw_df: pd.DataFrame,
        train_seasons: list[str] | None = None,
        val_seasons: list[str] | None = None,
        test_seasons: list[str] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_seasons = train_seasons or ["2018-19", "2019-20", "2020-21", "2021-22"]
        val_seasons = val_seasons or ["2022-23"]
        test_seasons = test_seasons or ["2023-24"]

        team_df = self._prepare_team_logs(raw_df)
        featured_df = self._compute_rolling_features(team_df)
        matchups = self._build_matchup_dataset(featured_df)

        train_df = matchups[matchups["SEASON"].isin(train_seasons)].copy().reset_index(drop=True)
        val_df = matchups[matchups["SEASON"].isin(val_seasons)].copy().reset_index(drop=True)
        test_df = matchups[matchups["SEASON"].isin(test_seasons)].copy().reset_index(drop=True)

        self.scaler.fit(train_df[self.feature_columns].values)

        train_df[self.feature_columns] = self.scaler.transform(train_df[self.feature_columns].values)
        val_df[self.feature_columns] = self.scaler.transform(val_df[self.feature_columns].values)
        test_df[self.feature_columns] = self.scaler.transform(test_df[self.feature_columns].values)

        self.paths.data_processed.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(self.paths.data_processed / "train.parquet", index=False)
        val_df.to_parquet(self.paths.data_processed / "val.parquet", index=False)
        test_df.to_parquet(self.paths.data_processed / "test.parquet", index=False)

        joblib.dump(self.scaler, self.paths.data_processed / "scaler.joblib")
        with open(self.paths.data_processed / "feature_names.json", "w") as f:
            json.dump(self.feature_columns, f, indent=2)

        return train_df, val_df, test_df
