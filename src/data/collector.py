import time
from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from src.utils.config import ProjectPaths

class NBADataCollector:
    def __init__(self, paths: ProjectPaths | None = None):
        self.paths = paths or ProjectPaths()
        self.raw_file = self.paths.data_raw / "nba_games_raw.parquet"

    def fetch_season(self, season: str, max_retries: int = 5, retry_delay: float = 2.0) -> pd.DataFrame:
        for attempt in range(max_retries):
            try:
                log = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star="Regular Season",
                    timeout=30
                )
                frames = log.get_data_frames()
                if frames and not frames[0].empty:
                    df = frames[0]
                    df["SEASON"] = season
                    return df
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay * (attempt + 1))
        raise RuntimeError(f"Failed to fetch season {season}")

    def collect_all_seasons(
        self,
        seasons: list[str] | None = None,
        force_download: bool = False
    ) -> pd.DataFrame:
        if self.raw_file.exists() and not force_download:
            return pd.read_parquet(self.raw_file)

        if seasons is None:
            seasons = [
                "2018-19",
                "2019-20",
                "2020-21",
                "2021-22",
                "2022-23",
                "2023-24"
            ]

        season_dfs: list[pd.DataFrame] = []
        for season in seasons:
            df = self.fetch_season(season)
            season_dfs.append(df)
            time.sleep(1.0)

        combined = pd.concat(season_dfs, ignore_index=True)
        combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"])
        combined = combined.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        self.paths.data_raw.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.raw_file, index=False)
        return combined
