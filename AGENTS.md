# AGENTS.md

This file gives pointers for people and coding agents that use CSIGen or contribute to it. The [README](README.md) explains what CSIGen does and how to run it. The manual `docs/main.pdf` describes the configuration and the outputs in full. This file does not repeat them. It tells you where to look and which conventions to follow.

## Where to start

| Task | Read |
|---|---|
| Install CSIGen and run a first config | README, sections "Installation" and "Generating channels from a config" |
| Find the config keys, their allowed values and their defaults | `docs/main.pdf`, section "Configuration", and the configs in `config/examples/` |
| Understand how a run works and what it writes | `docs/main.pdf`, sections "How a Run Works" and "Output" |
| Read saved path data or rebuild the CFR | README section "Path data", `docs/main.pdf`, section "Path Data", and `examples/CSIGen/rebuild_cfr_from_paths.ipynb` |
| Split a run into train and test | README section "Splitting a run into train and test" |
| Find the changes between versions | `docs/changelog.md` |

## Set up and test

CSIGen needs Python 3.12 or later. The tests need neither Sionna RT nor a GPU, so the base install with the test extra is enough.

```bash
pip install -e ".[test]"
pytest
```

The tests run on the CPU on small synthetic runs. To run one file or one test, give its path or a part of its name.

```bash
pytest tests/test_split.py -k disjoint
```

The CI (`.github/workflows/tests.yml`) runs `pytest` on Python 3.12 for every pull request and every push to `main`. It runs once with the tested versions of the README and once with the newest versions that pip resolves.

To generate channels, install the extra `rt` (`pip install -e ".[rt]"`) and run `csigen-run` from the repository root. A run needs Sionna RT, usually a GPU, and can take a long time. `pytest` checks only the modules that work without Sionna RT. It does not run the generation modules, such as `channel_generator.py`, `path_solver.py`, `radio_map.py` and `channel.py`. A change to a generation module needs a short run of an example config on a GPU. Ask before you start such a run, and say in the PR whether you did it.

To rebuild `docs/main.pdf`, run `./scripts/compile_docs.sh`. It needs a LaTeX install with the package `minted`, which runs Pygments through `-shell-escape`.

## How the code is organized

| Module | Contents |
|---|---|
| `csigen/__init__.py` | The version and the public names. It imports each module when it is first used |
| `csigen/config_validator.py` | The config schema, its defaults and its checks |
| `csigen/channel_generator.py` | `generate_channels`, which runs a config end to end and yields the data of each sector |
| `csigen/scene_setup.py`, `csigen/utils.py` | The load and preparation of a scene, and the geometry helpers |
| `csigen/base_station.py` | The base stations and their antenna arrays |
| `csigen/radio_map.py` | The radio map and the sampling of the user positions |
| `csigen/receivers.py`, `csigen/user_equipment.py` | The receivers, and the orientations and velocities of the users |
| `csigen/path_solver.py` | The path solver, in batches of receivers for each TX |
| `csigen/channel.py` | The CFR that Sionna RT computes, and the save of the output files |
| `csigen/path_data.py` | The saved paths, the CFR computed from them (`cfr_from_paths`, `load_cfr`) and the path statistics |
| `csigen/split.py` | The train and test splits of a run |
| `csigen/cli.py` | The commands `csigen-run` and `csigen-split` that `pyproject.toml` installs |

## Rules for changes

These rules keep the generated data reproducible and the saved data readable between versions.

- **Keep the base install light.** `csigen/path_data.py`, `csigen/split.py`, `csigen/config_validator.py` and `csigen/cli.py` need only NumPy, PyYAML and pydantic. Do not import Sionna RT, Mitsuba, Dr.Jit or matplotlib in them, or at the top level of `csigen/__init__.py`. `tests/test_package.py` checks this.
- **Keep the defaults.** Most keys are required. The optional keys have defaults in `csigen/config_validator.py`, and these defaults set the data of every config that leaves out such a key. A change of a default changes the generated data. Discuss such a change in an issue first.
- **Keep the data stable.** If a change can change the channels, the users, the positions or the metadata of a run, say so in the PR and show the size of the change. A refactor or a speed-up must give the same data.
- **Keep the CFR exact.** With the NumPy version and the CPU of the generation, `load_cfr` rebuilds the saved CFR bit for bit. `tests/test_path_data.py` checks the bits on a synthetic run. It also checks the CFR against `tests/data/golden_cfr.npz` to within a few float32 steps, because the rounding can change with another CPU or NumPy build. Do not make the golden file again to make a test pass.
- **Keep old runs readable.** `load_cfr` and `csigen-split` read the files of earlier versions. When you change the output files or the metadata, keep the reading of the old files.
- **Add a config option in all places.** A new key needs a field and its checks in `csigen/config_validator.py`, a description in the section "Configuration" of `docs/main.tex` and a test in `tests/test_config.py`. Add a required key also to the complete example config in that section and to every config in `config/examples/`, because `tests/test_config.py` validates them. Rebuild `docs/main.pdf` with `./scripts/compile_docs.sh`.

## Code style

- Put module-level constants at the top of each file, after the imports and the logger.
- Add a test in `tests/` with each new module or behavior, and run `pytest` before you open a pull request.
- Do not commit generated runs, path data or other large outputs. `output/` and `*.npz` files are ignored by git, except the small files in `tests/data/`.
- Match the style of the code around your change. Change only the lines that your change needs.

## Pull requests

- Open an issue first for a large change, a new config option or a change of a default.
- Name the branch `feat/<topic>`, `fix/<topic>` or `docs/<topic>`.
- Write commit messages with a short imperative summary line, then a few lines on what changed and why.
- One pull request covers one topic. Use a short imperative title.
- Fill in `.github/pull_request_template.md` (Why, Where, How). In "How", say how you checked the change, for example "pytest passes".
- Do not change the version in `csigen/__init__.py` or `docs/changelog.md`. `test_version_matches_the_changelog` requires the two to agree. The maintainer makes the releases and writes the changelog from the descriptions of the merged pull requests. Therefore, describe in your pull request what a user of CSIGen can notice, for example a change of the generated data, a new config option or a new output file.

## Writing style

The docs, docstrings, comments, commit messages and pull requests use simple technical English.

- Write short sentences, with one idea per sentence, in the active voice and the present tense.
- Use simple words and American English.
