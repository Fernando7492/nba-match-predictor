import pytest
import numpy as np
import pandas as pd
from src.data.advanced_features import (
    compute_four_factors_and_pace,
    compute_elo_ratings,
    compute_head_to_head_features,
    ADVANCED_STAT_COLS
)
from src.data.preprocessor import NBAPreprocessor
from src.data.collector import NBADataCollector

def test_four_factors_computation_bounds():
    df = pd.DataFrame({
        "TEAM_ID": [1, 1],
        "GAME_ID": ["G1", "G2"],
        "GAME_DATE": pd.to_datetime(["2023-01-01", "2023-01-03"]),
        "PTS": [110, 95],
        "PTS_ALLOWED": [105, 100],
        "FGM": [40, 35],
        "FGA": [85, 80],
        "FG3M": [12, 10],
        "FTM": [18, 15],
        "FTA": [22, 20],
        "OREB": [10, 8],
        "DREB": [32, 30],
        "TOV": [14, 12]
    })
    
    out = compute_four_factors_and_pace(df)
    assert all(col in out.columns for col in ADVANCED_STAT_COLS)
    assert (out["EFG_PCT"] >= 0.0).all() and (out["EFG_PCT"] <= 1.0).all()
    assert (out["TOV_PCT"] >= 0.0).all() and (out["TOV_PCT"] <= 1.0).all()
    assert (out["POSS"] >= 60.0).all() and (out["POSS"] <= 140.0).all()

def test_elo_ratings_evolution_and_properties():
    df = pd.DataFrame({
        "GAME_ID": ["G1", "G2"],
        "GAME_DATE": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "SEASON": ["2022-23", "2022-23"],
        "TEAM_ID_HOME": [1, 2],
        "TEAM_ID_AWAY": [2, 1],
        "PTS_HOME": [110, 100],
        "PTS_AWAY": [100, 105],
        "TARGET_HOME_W": [1, 0]
    })
    
    out = compute_elo_ratings(df)
    assert "ELO_HOME" in out.columns
    assert "ELO_AWAY" in out.columns
    assert "ELO_EXPECTED_PROB" in out.columns
    assert (out["ELO_EXPECTED_PROB"] >= 0.0).all() and (out["ELO_EXPECTED_PROB"] <= 1.0).all()

def test_head_to_head_features():
    df = pd.DataFrame({
        "GAME_ID": ["G1", "G2"],
        "GAME_DATE": pd.to_datetime(["2023-01-01", "2023-01-10"]),
        "TEAM_ID_HOME": [1, 2],
        "TEAM_ID_AWAY": [2, 1],
        "PTS_HOME": [100, 90],
        "PTS_AWAY": [90, 95],
        "TARGET_HOME_W": [1, 0]
    })
    
    out = compute_head_to_head_features(df)
    assert "H2H_WIN_RATE" in out.columns
    assert out["H2H_WIN_RATE"].iloc[0] == 0.5
    assert out["H2H_WIN_RATE"].iloc[1] == 0.0

def test_preprocessor_with_advanced_features_integration():
    collector = NBADataCollector()
    raw_df = collector.collect_all_seasons(seasons=["2022-23", "2023-24"])
    
    preprocessor = NBAPreprocessor()
    train_df, val_df, test_df = preprocessor.process_and_split(
        raw_df, train_seasons=["2022-23"], val_seasons=["2022-23"], test_seasons=["2023-24"]
    )
    
    assert len(preprocessor.feature_columns) > 195
    assert "ELO_DIFF" in preprocessor.feature_columns
    assert "H2H_WIN_RATE" in preprocessor.feature_columns
    assert not test_df[preprocessor.feature_columns].isnull().any().any()
