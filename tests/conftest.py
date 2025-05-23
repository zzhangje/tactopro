import pytest
from tactopro import Config


@pytest.fixture(scope="session")
def config():
    cfg = Config()
    cfg.headless = True
    return cfg
