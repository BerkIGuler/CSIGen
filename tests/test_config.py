import pytest
import yaml

from helpers import REPO
from src.config_validator import load_validated_config, validate_config

EXAMPLES = sorted((REPO / "config" / "examples").rglob("*.yaml"))


def test_examples_are_found():
    assert {"urban", "suburban", "rural"} <= {p.parent.name for p in EXAMPLES}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_example_config_is_valid(path, monkeypatch):
    monkeypatch.chdir(REPO)                         # scene paths are relative to the repo root
    config = load_validated_config(path)
    assert (REPO / config["scene_xml_path"]).is_file()


def test_missing_scene_is_rejected(monkeypatch):
    monkeypatch.chdir(REPO)
    config = yaml.safe_load(EXAMPLES[0].read_text())
    config["scene_xml_path"] = "scenes/no_such_scene/scene.xml"
    with pytest.raises(ValueError, match="no_such_scene"):
        validate_config(config)
