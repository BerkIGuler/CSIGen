import pytest

from helpers import write_run


@pytest.fixture
def run_dir(tmp_path):
    """A small synthetic run (see helpers.write_run)."""
    return write_run(tmp_path / "run")
