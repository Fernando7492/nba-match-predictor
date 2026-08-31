import time
from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from src.utils.config import ProjectPaths, DEFAULT_SEASONS_ALL

class NBADataCollector:
    def __init__(self, paths: ProjectPaths | None = None):
        self.paths = paths or ProjectPaths()

    def fetch_season_gamelogs(
        self,
        season: str,
        season_type: str = "Regular Season",
        max_retries: int = 3,
        delay: float = 0.8
    ) -> pd.DataFrame:
        for attempt in range(max_retries):
            try:
                time.sleep(delay)
                log = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star=season_type,
                    player_or_team_abbreviation="T",
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
                time.sleep(delay * (attempt + 1))
        return pd.DataFrame()

    def collect_all_seasons(
        self,
        seasons: list[str] | None = None,
        force_download: bool = False
    ) -> pd.DataFrame:
        seasons = seasons or DEFAULT_SEASONS_ALL

        self.paths.data_raw.mkdir(parents=True, exist_ok=True)
        cache_file = self.paths.data_raw / "nba_raw_gamelogs.parquet"

        if cache_file.exists() and not force_download:
            cached_df = pd.read_parquet(cache_file)
            available_seasons = set(cached_df["SEASON"].unique())
            if set(seasons).issubset(available_seasons):
                return cached_df[cached_df["SEASON"].isin(seasons)].copy()

        all_frames = []
        for season in seasons:
            df = self.fetch_season_gamelogs(season=season)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            raise RuntimeError("No data was collected from NBA API.")

        full_df = pd.concat(all_frames, ignore_index=True)
        full_df.to_parquet(cache_file, index=False)
        return full_df
