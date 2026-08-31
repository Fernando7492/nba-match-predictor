import pytest
import pandas as pd
import numpy as np
import torch
from src.data.collector import NBADataCollector
from src.data.preprocessor import NBAPreprocessor
from src.data.dataset import NBATabularDataset, get_tabular_loaders
from src.utils.config import ProjectPaths

@pytest.fixture(scope="module")
def processed_data():
    paths = ProjectPaths()
    collector = NBADataCollector(paths=paths)
    raw_df = collector.collect_all_seasons()
    preprocessor = NBAPreprocessor(paths=paths)
    train_df, val_df, test_df = preprocessor.process_and_split(raw_df)
    return train_df, val_df, test_df, preprocessor.feature_columns

def test_raw_data_schema():
    paths = ProjectPaths()
    raw_path = paths.data_raw / "nba_games_raw.parquet"
    assert raw_path.exists()
    df = pd.read_parquet(raw_path)
    required_cols = ["GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP", "WL", "PTS"]
    for col in required_cols:
        assert col in df.columns

def test_no_null_features(processed_data):
    train_df, val_df, test_df, features = processed_data
    assert train_df[features].isnull().sum().sum() == 0
    assert val_df[features].isnull().sum().sum() == 0
    assert test_df[features].isnull().sum().sum() == 0

def test_target_binary_nature(processed_data):
    train_df, val_df, test_df, _ = processed_data
    for df in [train_df, val_df, test_df]:
        targets = df["TARGET_HOME_W"].unique()
        assert set(targets).issubset({0, 1})

def test_strict_temporal_split(processed_data):
    train_df, val_df, test_df, _ = processed_data
    max_train_date = pd.to_datetime(train_df["GAME_DATE"]).max()
    min_val_date = pd.to_datetime(val_df["GAME_DATE"]).min()
    max_val_date = pd.to_datetime(val_df["GAME_DATE"]).max()
    min_test_date = pd.to_datetime(test_df["GAME_DATE"]).min()

    assert max_train_date < min_val_date
    assert max_val_date < min_test_date

def test_rolling_no_future_leakage():
    sample_data = pd.DataFrame([
        {"GAME_ID": "1", "GAME_DATE": "2023-01-01", "TEAM_ID": 101, "MATCHUP": "BOS vs. MIA", "WL": "W", "PTS": 120, "FGM": 40, "FGA": 80, "FG_PCT": 0.5, "FG3M": 10, "FG3A": 30, "FG3_PCT": 0.33, "FTM": 30, "FTA": 35, "FT_PCT": 0.85, "OREB": 10, "DREB": 30, "REB": 40, "AST": 25, "STL": 8, "BLK": 5, "TOV": 10, "PF": 18, "PLUS_MINUS": 20, "SEASON": "2022-23"},
        {"GAME_ID": "1", "GAME_DATE": "2023-01-01", "TEAM_ID": 102, "MATCHUP": "MIA @ BOS", "WL": "L", "PTS": 100, "FGM": 35, "FGA": 80, "FG_PCT": 0.43, "FG3M": 8, "FG3A": 28, "FG3_PCT": 0.28, "FTM": 22, "FTA": 28, "FT_PCT": 0.78, "OREB": 8, "DREB": 28, "REB": 36, "AST": 20, "STL": 6, "BLK": 4, "TOV": 12, "PF": 20, "PLUS_MINUS": -20, "SEASON": "2022-23"},
        {"GAME_ID": "2", "GAME_DATE": "2023-01-03", "TEAM_ID": 101, "MATCHUP": "BOS vs. NYK", "WL": "W", "PTS": 110, "FGM": 38, "FGA": 82, "FG_PCT": 0.46, "FG3M": 12, "FG3A": 32, "FG3_PCT": 0.37, "FTM": 22, "FTA": 26, "FT_PCT": 0.84, "OREB": 9, "DREB": 31, "REB": 40, "AST": 24, "STL": 7, "BLK": 6, "TOV": 9, "PF": 17, "PLUS_MINUS": 10, "SEASON": "2022-23"},
        {"GAME_ID": "2", "GAME_DATE": "2023-01-03", "TEAM_ID": 103, "MATCHUP": "NYK @ BOS", "WL": "L", "PTS": 100, "FGM": 36, "FGA": 85, "FG_PCT": 0.42, "FG3M": 9, "FG3A": 30, "FG3_PCT": 0.30, "FTM": 19, "FTA": 24, "FT_PCT": 0.79, "OREB": 10, "DREB": 29, "REB": 39, "AST": 21, "STL": 5, "BLK": 3, "TOV": 11, "PF": 19, "PLUS_MINUS": -10, "SEASON": "2022-23"},
    ])
    preprocessor = NBAPreprocessor(rolling_windows=[3])
    team_df = preprocessor._prepare_team_logs(sample_data)
    featured = preprocessor._compute_rolling_features(team_df)
    bos_first_game = featured[(featured["TEAM_ID"] == 101) & (featured["GAME_ID"] == "1")]
    bos_second_game = featured[(featured["TEAM_ID"] == 101) & (featured["GAME_ID"] == "2")]
    assert pd.isna(bos_first_game["PTS_ROLL_3"].values[0]) or np.isnan(bos_first_game["PTS_ROLL_3"].values[0])
    assert bos_second_game["PTS_ROLL_3"].values[0] == pytest.approx(120.0)

def test_dataloader_batch_shapes(processed_data):
    train_df, val_df, test_df, features = processed_data
    train_loader, val_loader, test_loader = get_tabular_loaders(
        train_df, val_df, test_df, features, batch_size=32
    )
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape == (32, len(features))
    assert y_batch.shape == (32, 1)
    assert x_batch.dtype == torch.float32
    assert y_batch.dtype == torch.float32
