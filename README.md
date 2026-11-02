# CSIGen

Config-driven, easy-to-use wireless channel dataset generator built on **Sionna RT**. Use the sections below in order, or jump to `docs/main.pdf` for the full manual.

## Generating channels from a config

After you define a YAML config (see examples under `config/` and `docs/main.pdf`), run:

```bash
python scripts/run.py --config path/to/your_config.yaml
```

Run from the repository root (or ensure Python can resolve the `src` package as in `scripts/run.py`). The script loads the config, streams channel outputs per transmit sector, and writes timestamped artifacts under `output/` (see `docs/main.pdf` for layout and metadata).

## Path data

With a synthetic array (`path_solver_synthetic_array: true`, as in the example configs), each file also holds the paths of every channel (delay, Doppler shift, angles of departure and arrival, coefficient per antenna pair, interaction types) and per-channel statistics such as the delay and angular spreads, the K-factor and the strongest path. The saved CFR is computed from these paths, so it can be rebuilt exactly, or for another numerology, without new ray tracing:

```python
from src.path_data import load_cfr
h = load_cfr("output/<scene_name>/<run_id>/channel_tx_0.npz")
```

`load_cfr` also takes another numerology (`num_subcarriers`, `subcarrier_spacing`, `num_ofdm_symbols`, `symbol_duration`), other planar arrays (`tx_array=dict(num_rows=8, num_cols=8)`), other RX velocities (`rx_velocities=[20.0, 0.0, 0.0]`) or only some of the paths (`max_paths=3`, `max_bounces=0` for LoS only, `power_range_db`, `exclude_interactions`). With `save_cfr: false` in the config, the files hold only the paths. See `docs/main.pdf` for the fields. `load_cfr` needs only NumPy, so you can use the path data without Sionna RT or a GPU. For this, install only NumPy instead of `requirements.txt`, and run from the repository root (or add it to `sys.path`) so that Python finds the `src` package.

This makes CSIGen datasets easy to share. With a few paths per channel, the paths typically take only a few percent of the size of the CFR. Therefore, share the files without `h` (generate with `save_cfr: false`, or remove `h` afterwards), and anyone can rebuild the exact CFR with `load_cfr`.

## Splitting a run into train and test

```bash
python scripts/split_dataset.py --run output/<scene_name>/<run_id> --out <out_dir> --by bs --test-ratio 0.2 --seed 1
```

This writes `<out_dir>/train/` and `<out_dir>/test/`, each a run in the same format, with a `split.yaml` that records the method, its parameters and the counts. `--by` selects random users (`users`), areas of the scene (`area`), sectors (`sector`), base stations (`bs`), or a sector or base-station split combined with a user split (for example `bs+area`). See `docs/main.pdf` for the options.

## Requirements

| Component | Version (reference) |
|-----------|---------------------|
| **Python** | **3.12.x** |
| **Sionna RT** | **2.2.0** |

Install dependencies into your environment:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins packages as resolved in our system. CSIGen uses only Sionna RT (`sionna-rt`), not the full Sionna package or TensorFlow. Follow [Sionna](https://github.com/NVlabs/sionna) install guidance if `pip install` fails.

We tested Sionna RT 2.2.0 on the GPU with NVIDIA driver 580, and it works.

### GPU acceleration

Throughput improves substantially when **Sionna RT / Mitsuba** runs with a supported **GPU** (and matching **CUDA/driver** toolchain). Misconfiguration can cause errors or quiet fallback to slower paths.

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
├── src/              # Library: scene setup, radio map / path solvers, path data, CFR, config validation
├── config/           # YAML configs; examples under examples/
│                     #   (urban/, suburban/ and rural/, for example urban/boston_3p5ghz.yaml)
├── scenes/           # Example scenes, each with <name>_1/scene.xml and its meshes. Five downtowns,
│                     #   two suburban (irvine_1, paloalto_1) and two rural (aguadulce_1, littlerock_1)
├── scripts/          # run.py, split_dataset.py, compile_docs.sh, and other CLI helpers
├── docs/             # main.tex and build script for the PDF manual (main.pdf); changelog.md
├── examples/         # Notebooks (paths assume repo root on sys.path like scripts/run.py)
│   ├── CSIGen/       # Notebooks that call run.py and visualize saved output
│   └── sionna/       # Smaller Sionna RT scene previews (empty / tutorial-style)
├── output/           # Timestamped channel runs: output/<scene_name>/<run_id>/ …
│                     #   (created by run.py; ignored by git by default—see .gitignore)
├── requirements.txt  # Pinned Python deps
├── LICENSE
└── README.md
```

## A sample dataset generated with this tool can be found [here](https://huggingface.co/datasets/BerkIGuler/ChaSEDatasets/).

It holds runs of eight scenes of `scenes/` at 3.5 GHz, all of them except `la_1`. The urban runs were made with CSIGen 0.3.4, and the suburban and rural runs with 0.3.6. The runs are in `runs/<scene type>/<scene>/`. The CFR `h` was removed from the files after generation to keep the shared files small. Rebuild it with `load_cfr` (see "Path data" above) if you need it. This dataset was generated to demonstrate a tool that we developed, [ChaSE](https://github.com/BerkIGuler/ChaSE), so it also holds a few other files, such as data splits.

## License

This software is licensed under the **MIT License**—see [`LICENSE`](LICENSE). The software is provided **“as is”**, without warranty of any kind. See the license file for the full disclaimer and terms.
