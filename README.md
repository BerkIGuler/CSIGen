# CSIGen

Config-driven, easy-to-use wireless channel dataset generator built on **Sionna RT**. Use the sections below in order, or jump to `docs/main.pdf` for the full manual.

## Installation

Clone the repository and install CSIGen with pip, from the repository root.

```bash
git clone https://github.com/BerkIGuler/CSIGen.git
cd CSIGen
pip install -e ".[rt]"
```

The extra `rt` adds Sionna RT and Mitsuba for channel generation. CSIGen uses only Sionna RT (`sionna-rt`), not the full Sionna package or TensorFlow. Follow the [Sionna](https://github.com/NVlabs/sionna) install guidance if `pip install` fails. To only read saved path data (see "Path data" below), validate configs or split runs, `pip install -e .` is enough. It needs only NumPy, PyYAML and pydantic, and it works without Sionna RT or a GPU. The extra `examples` adds Jupyter and matplotlib for the notebooks in `examples/`. Only `rebuild_cfr_from_paths.ipynb` works without Sionna RT. The other notebooks also need the extra `rt`, so install `pip install -e ".[rt,examples]"` for them.

CSIGen needs Python 3.12 or later and is tested with these versions.

| Component | Version (reference) |
|-----------|---------------------|
| **Python** | **3.12** |
| **Sionna RT** | **2.2.0** |
| **Mitsuba** | 3.9.1 |
| **NumPy** | 2.5.3 |
| **PyYAML** | 6.0.3 |
| **pydantic** | 2.14.0 |
| **matplotlib** | 3.11.2 |

The extra `rt` pins Sionna RT and Mitsuba to the tested versions. The other dependencies in `pyproject.toml` are lower bounds, so pip can install newer versions. To install the tested versions of all packages, pin the others in the install command.

```bash
pip install -e ".[rt]" numpy==2.5.3 PyYAML==6.0.3 pydantic==2.14.0 matplotlib==3.11.2
```

With another NumPy version or another CPU, `load_cfr` can give a CFR that differs from the saved one in the last bits.

Sionna RT needs a recent NVIDIA driver with CUDA support to run on the GPU.

### GPU acceleration

Throughput improves substantially when **Sionna RT / Mitsuba** runs with a supported **GPU** (and matching **CUDA/driver** toolchain). Misconfiguration can cause errors or quiet fallback to slower paths.

### Tests

The tests check the path data, `load_cfr`, the config validation, the splits and the commands on small synthetic runs. They need neither Sionna RT nor a GPU.

```bash
pip install -e ".[test]"
pytest
```

On GitHub, the workflow `.github/workflows/tests.yml` runs the tests for each pull request and each push to `main`, once with the tested versions above and once with the newest versions that pip resolves.

## Generating channels from a config

After you define a YAML config (see examples under `config/` and `docs/main.pdf`), run the command below.

```bash
csigen-run --config path/to/your_config.yaml
```

Run it from the repository root, because the example configs give the scene files relative to it. The command loads the config, streams channel outputs per transmit sector, and writes timestamped artifacts under `output/` in the working directory (see `docs/main.pdf` for layout and metadata).

## Path data

With a synthetic array (`path_solver_synthetic_array: true`, as in the example configs), each file also holds the paths of every channel (delay, Doppler shift, angles of departure and arrival, coefficient per antenna pair, interaction types) and per-channel statistics such as the delay and angular spreads, the K-factor and the strongest path. The saved CFR is computed from these paths, so it can be rebuilt (see below), or for another OFDM numerology, without new ray tracing:

```python
from csigen.path_data import load_cfr
h = load_cfr("output/<scene_name>/<run_id>/channel_tx_0.npz")
```

`load_cfr` also takes another numerology (`num_subcarriers`, `subcarrier_spacing`, `num_ofdm_symbols`, `symbol_duration`), other planar arrays (`tx_array=dict(num_rows=8, num_cols=8)`), other RX velocities (`rx_velocities=[20.0, 0.0, 0.0]`) or only some of the paths (`max_paths=3`, `max_bounces=0` for LoS only, `power_range_db`, `exclude_interactions`). With `save_cfr: false` in the config, the files output by `csigen-run` hold only the paths and `load_cfr()` is needed to build the CFR. See `docs/main.pdf` for the fields.

This makes CSIGen datasets easy to share. With a few paths per channel, the paths typically take only a few percent of the size of the CFR. Therefore, you may want to share the files without `h` (generate with `save_cfr: false`, or remove `h` afterwards), and anyone can rebuild the CFR with `load_cfr`. With the NumPy version and CPU type of the generation, it is the saved CFR bit for bit.

## Splitting a run into train and test

```bash
csigen-split --run output/<scene_name>/<run_id> --out <out_dir> --by bs --test-ratio 0.2 --seed 1
```

This writes `<out_dir>/train/` and `<out_dir>/test/`, each a run in the same format, with a `split.yaml` that records the method, its parameters and the counts. `--by` selects random users (`users`), areas of the scene (`area`), sectors (`sector`), base stations (`bs`), or a sector or base-station split combined with a user split (for example `bs+area`). See `docs/main.pdf` for the options.

## Documentation

Read `docs/main.pdf` for:

- Full **YAML configuration reference** (every parameter and allowed values).
- How **scene generation** (Geo2SigMap / `scenegen`) ties into CSIGen.
- How **channel generation** runs end-to-end and what assumptions the pipeline makes (measurement surface, BS placement, sampling, LoS labeling, outputs, etc.).

Rebuild the PDF manual from source if needed:

```bash
./scripts/compile_docs.sh
```

## Purpose and scope

CSIGen is a **pipeline for producing large-scale wireless channel datasets** aimed at **training and evaluating wireless physical-layer AI models**. It builds on NVIDIA’s **Sionna** ray-tracing primitives but exposes a **single config file** to drive scene setup, sampling, solvers, and CSI export so you can scale generation without touching core code for each experiment. Configuration is documented in depth in `docs/main.pdf`.

This project is released **publicly** to help accelerate **wireless AI research**. **Feedback, issues, and fixes are welcome.** If you need a new feature or have a question about using CSIGen, please [open an issue](https://github.com/BerkIGuler/CSIGen/issues).

## Repository layout

```
├── csigen/           # Package: scene setup, radio map / path solvers, path data, CFR, config validation, commands
├── config/           # YAML configs; examples under examples/
│                     #   (urban/, suburban/ and rural/, for example urban/boston_3p5ghz.yaml)
├── scenes/           # Example scenes, each with <name>_1/scene.xml and its meshes. Five downtowns,
│                     #   two suburban (irvine_1, paloalto_1) and two rural (aguadulce_1, littlerock_1)
├── scripts/          # compile_docs.sh, which builds the PDF manual
├── docs/             # main.tex and build script for the PDF manual (main.pdf); changelog.md
├── tests/            # Tests that run without Sionna RT or a GPU (pytest)
├── examples/         # Notebooks (run them with the repository root or a folder inside it as the working directory)
│   ├── CSIGen/       # Notebooks that generate and plot small runs, and one that rebuilds the CFR from saved path data
│   └── sionna/       # Smaller Sionna RT scene previews (empty / tutorial-style)
├── output/           # Timestamped channel runs: output/<scene_name>/<run_id>/ …
│                     #   (created by csigen-run; ignored by git by default, see .gitignore)
├── pyproject.toml    # Package, dependencies, extras, commands and test settings
├── LICENSE
└── README.md
```

## A sample dataset generated with this tool can be found [here](https://huggingface.co/datasets/BerkIGuler/ChaSEDatasets/).

The sample dataset holds runs of eight scenes of `scenes/` at 3.5 GHz, all of them except `la_1`. The urban runs were made with CSIGen 0.3.4, and the suburban and rural runs with 0.3.6. The runs are in `runs/<scene type>/<scene>/`. The CFR `h` was removed from the files after generation to keep the shared files small. Rebuild it with `load_cfr` (see "Path data" above, and `examples/CSIGen/rebuild_cfr_from_paths.ipynb` for a worked example on one file of this dataset) if you need it. This dataset was generated to demonstrate a tool that we developed, [ChaSE](https://github.com/BerkIGuler/ChaSE), so it also holds a few other files, such as data splits.

## License

This software is licensed under the **MIT License**—see [`LICENSE`](LICENSE). The software is provided **“as is”**, without warranty of any kind. See the license file for the full disclaimer and terms.
