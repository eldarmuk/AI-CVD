import pytest
import torch

from ai_cvd.config import public_configuration
from ai_cvd.data.synthetic import build_fixture


@pytest.fixture(scope="session", autouse=True)
def deterministic_torch():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)


@pytest.fixture(scope="session")
def fixture():
    config = public_configuration()
    return (config, *build_fixture(config))
