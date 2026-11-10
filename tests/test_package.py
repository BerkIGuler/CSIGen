import re
import subprocess
import sys

import pytest

from helpers import REPO, command

# Packages of the ray tracing that the reading of saved data must not need
HEAVY = ("sionna", "mitsuba", "drjit", "tensorflow", "matplotlib")
# Modules of csigen that need only numpy, pyyaml and pydantic
LIGHT_MODULES = ("csigen.path_data", "csigen.split", "csigen.config_validator")

NO_HEAVY = f"""
import sys
for name in {HEAVY!r}:
    sys.modules[name] = None        # any import of these raises ImportError
import csigen
from csigen import load_cfr, validate_config, path_stats
from csigen.split import write_split
try:
    csigen.generate_channels
except AttributeError as err:
    assert "needs a package that is not installed" in str(err)
else:
    raise AssertionError("generate_channels imported without Sionna RT")
print("ok")
"""


def test_saved_data_and_configs_work_without_sionna():
    out = subprocess.run([sys.executable, "-c", NO_HEAVY], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


@pytest.mark.parametrize("main", ["run_main", "split_main"])
def test_command_help_works_without_sionna(main):
    code = (f"import sys\nfor name in {HEAVY!r}:\n    sys.modules[name] = None\n"
            f"from csigen.cli import {main}\n{main}(['--help'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "--" in out.stdout


def test_commands_are_installed():
    for name in ("csigen-run", "csigen-split"):
        out = subprocess.run([command(name), "--help"], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr


def test_light_exports_resolve():
    import csigen
    for name, module in csigen._EXPORTS.items():
        if module in LIGHT_MODULES:
            assert getattr(csigen, name).__module__ == module


def test_version_matches_the_changelog():
    import csigen
    first = re.search(r"^## \[(\d+\.\d+\.\d+)\]", (REPO / "docs" / "changelog.md").read_text(), re.M)
    assert first, "docs/changelog.md has no section '## [X.Y.Z] - date'"
    assert csigen.__version__ == first.group(1)
