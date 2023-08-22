import pytest
from check import enable_debug


@pytest.fixture(scope="function", autouse=True)
def enable_checking():
    enable_debug()
