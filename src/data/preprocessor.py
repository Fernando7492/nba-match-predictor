import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from src.utils.config import (
    ProjectPaths,
    DEFAULT_SEASONS_TRAIN,
    DEFAULT_SEASONS_VAL,
    DEFAULT_SEASONS_TEST,
    BASE_STAT_COLS
)
from src.data.common import prepare_raw_team_logs
from src.data.advanced_features import (
    ADVANCED_STAT_COLS,
    compute_four_factors_and_pace,
    compute_elo_ratings,
    compute_head_to_head_features
)

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
        self.all_stat_cols = list(BASE_STAT_COLS) + list(ADVANCED_STAT_COLS)

    def _prepare_team_logs(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = prepare_raw_team_logs(raw_df)
        df = compute_four_factors_and_pace(df)
        return df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    def _compute_rolling_features(self, team_df: pd.DataFrame) -> pd.DataFrame:
        grouped = team_df.groupby("TEAM_ID")
        rest_days = grouped["GAME_DATE"].diff().dt.total_seconds() / 86400.0
        rest_days = rest_days.fillna(7.0).clip(lower=0.0, upper=10.0)
        back_to_back = (rest_days <= 1.0).astype(float)

        new_cols: dict[str, pd.Series] = {
            "REST_DAYS": rest_days,
            "BACK_TO_BACK": back_to_back
        }

        for col in self.all_stat_cols:
            shifted = grouped[col].shift(1)
            for w in self.rolling_windows:
                roll = shifted.groupby(team_df["TEAM_ID"]).rolling(window=w, min_periods=1).mean()
                new_cols[f"{col}_ROLL_{w}"] = roll.reset_index(level=0, drop=True)

        rolling_df = pd.DataFrame(new_cols, index=team_df.index)
        return pd.concat([team_df, rolling_df], axis=1)

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
        matchups["GAME_DATE"] = pd.to_datetime(matchups["GAME_DATE_HOME"])
        matchups["SEASON"] = matchups["SEASON_HOME"]

        rolling_cols = [
            col for col in home.columns if "_ROLL_" in col or col in ["REST_DAYS", "BACK_TO_BACK"]
        ]

        diff_dict = {}
        diff_cols = []
        for col in rolling_cols:
            home_col = f"{col}_HOME"
            away_col = f"{col}_AWAY"
            diff_col = f"{col}_DIFF"
            diff_dict[diff_col] = matchups[home_col] - matchups[away_col]
            diff_cols.append(diff_col)

        matchups = pd.concat([matchups, pd.DataFrame(diff_dict, index=matchups.index)], axis=1)

        matchups = compute_elo_ratings(matchups)
        matchups = compute_head_to_head_features(matchups)

        elo_h2h_cols = [
            "ELO_HOME", "ELO_AWAY", "ELO_DIFF", "ELO_EXPECTED_PROB",
            "H2H_WIN_RATE", "H2H_POINT_DIFF", "H2H_GAMES_COUNT"
        ]

        feature_cols = (
            [f"{c}_HOME" for c in rolling_cols]
            + [f"{c}_AWAY" for c in rolling_cols]
            + diff_cols
            + elo_h2h_cols
        )

        matchups = matchups.dropna(subset=feature_cols)
        matchups = matchups.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        self.feature_columns = feature_cols
        return matchups

    def process_and_split(
        self,
        raw_df: pd.DataFrame,
        train_seasons: list[str] | None = None,
        val_seasons: list[str] | None = None,
        test_seasons: list[str] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_seasons = train_seasons or DEFAULT_SEASONS_TRAIN
        val_seasons = val_seasons or DEFAULT_SEASONS_VAL
        test_seasons = test_seasons or DEFAULT_SEASONS_TEST

        team_df = self._prepare_team_logs(raw_df)
        featured_df = self._compute_rolling_features(team_df)
        matchups = self._build_matchup_dataset(featured_df)

        train_df = matchups[matchups["SEASON"].isin(train_seasons)].copy().sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        val_df = matchups[matchups["SEASON"].isin(val_seasons)].copy().sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        test_df = matchups[matchups["SEASON"].isin(test_seasons)].copy().sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

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
