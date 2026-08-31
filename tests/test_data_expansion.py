import pytest
import pandas as pd
from src.data.collector import NBADataCollector
from src.data.preprocessor import NBAPreprocessor
from src.utils.config import ProjectPaths

def test_expanded_seasons_count():
    paths = ProjectPaths()
    collector = NBADataCollector(paths=paths)
    raw_df = collector.collect_all_seasons()
    
    assert raw_df["SEASON"].nunique() == 10
    assert len(raw_df) > 20000

def test_expanded_preprocessing_shapes():
    paths = ProjectPaths()
    collector = NBADataCollector(paths=paths)
    raw_df = collector.collect_all_seasons()
    preprocessor = NBAPreprocessor(paths=paths)
    train_df, val_df, test_df = preprocessor.process_and_split(raw_df)

    assert len(train_df) > 9000
    assert len(val_df) == 1230
    assert len(test_df) == 1230
    assert train_df[preprocessor.feature_columns].isnull().sum().sum() == 0

def test_temporal_non_leakage_expansion():
    paths = ProjectPaths()
    collector = NBADataCollector(paths=paths)
    raw_df = collector.collect_all_seasons()
    preprocessor = NBAPreprocessor(paths=paths)
    train_df, val_df, test_df = preprocessor.process_and_split(raw_df)

    max_train = pd.to_datetime(train_df["GAME_DATE"]).max()
    min_val = pd.to_datetime(val_df["GAME_DATE"]).min()
    max_val = pd.to_datetime(val_df["GAME_DATE"]).max()
    min_test = pd.to_datetime(test_df["GAME_DATE"]).min()

    assert max_train < min_val
    assert max_val < min_test
