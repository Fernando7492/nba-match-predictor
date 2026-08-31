import pytest
import torch
import numpy as np
import pandas as pd
import sklearn
import nba_api
from src.utils.config import ProjectPaths, ModelConfig
from src.utils.seed import set_seed

def test_imports():
    assert torch.__version__ is not None
    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None
    assert nba_api.__version__ is not None

def test_cuda_availability():
    assert torch.cuda.is_available() is True
    device_count = torch.cuda.device_count()
    assert device_count >= 1
    device_name = torch.cuda.get_device_name(0)
    assert len(device_name) > 0

def test_tensor_gpu_operations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    z = torch.matmul(x, y)
    assert z.item() == pytest.approx(32.0)

def test_seed_determinism():
    set_seed(42)
    rand1 = torch.rand(5)
    set_seed(42)
    rand2 = torch.rand(5)
    assert torch.equal(rand1, rand2)

def test_paths_structure():
    paths = ProjectPaths()
    assert paths.root.exists()
    assert paths.data_raw.exists()
    assert paths.data_processed.exists()
    assert paths.outputs_models.exists()
    assert paths.outputs_figures.exists()
