from dataclasses import dataclass, field
from pathlib import Path
import torch

DEFAULT_SEASONS_ALL: list[str] = [
    "2014-15", "2015-16", "2016-17", "2017-18",
    "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24"
]

DEFAULT_SEASONS_TRAIN: list[str] = [
    "2014-15", "2015-16", "2016-17", "2017-18",
    "2018-19", "2019-20", "2020-21", "2021-22"
]

DEFAULT_SEASONS_VAL: list[str] = ["2022-23"]
DEFAULT_SEASONS_TEST: list[str] = ["2023-24"]

BASE_STAT_COLS: list[str] = [
    "PTS", "PTS_ALLOWED", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
    "PLUS_MINUS", "WIN"
]

@dataclass
class ProjectPaths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)
    outputs_models: Path = field(init=False)
    outputs_figures: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.data_raw = self.root / "data" / "raw"
        self.data_processed = self.root / "data" / "processed"
        self.outputs_models = self.root / "outputs" / "models"
        self.outputs_figures = self.root / "outputs" / "figures"

@dataclass
class ModelConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 12
    sequence_length: int = 10
    rolling_windows: list[int] = field(default_factory=lambda: [3, 7, 14])
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
