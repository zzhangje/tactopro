import pytest
from tactopro import TactoConfig


@pytest.fixture(scope="session")
def config():
    cfg = TactoConfig()
    cfg.headless = True
    return cfg
