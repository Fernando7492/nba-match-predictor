from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(__file__).resolve().parent.parent.parent
    data_raw: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "raw")
    data_processed: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "processed")
    outputs_models: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "outputs" / "models")
    outputs_figures: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "outputs" / "figures")

@dataclass
class ModelConfig:
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 150
    early_stopping_patience: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    rolling_windows: list[int] = field(default_factory=lambda: [3, 7, 14])
    sequence_length: int = 10
