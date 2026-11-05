import re
import subprocess
import sys

from helpers import REPO

# Packages of the ray tracing that the reading of saved data must not need
HEAVY = ("sionna", "mitsuba", "drjit", "tensorflow", "matplotlib")
# Modules of src that need only numpy, pyyaml and pydantic
LIGHT_MODULES = ("src.path_data", "src.split", "src.config_validator")

NO_HEAVY = f"""
import sys
for name in {HEAVY!r}:
    sys.modules[name] = None        # any import of these raises ImportError
import src
from src import load_cfr, validate_config, path_stats
from src.split import write_split
try:
    src.generate_channels
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


def test_light_exports_resolve():
    import src
    for name, module in src._EXPORTS.items():
        if module in LIGHT_MODULES:
            assert getattr(src, name).__module__ == module


def test_version_matches_the_changelog():
    import src
    first = re.search(r"^## \[(\d+\.\d+\.\d+)\]", (REPO / "docs" / "changelog.md").read_text(), re.M)
    assert first, "docs/changelog.md has no section '## [X.Y.Z] - date'"
    assert src.__version__ == first.group(1)
