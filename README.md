# CSIGen

Config-driven wireless channel dataset generation built on **Sionna RT**. This README is short on purpose. Use the sections below in order. Or, jump to **`docs/main.pdf`** for the full manual.

## A sample dataset generated with this tool can be found [here](https://huggingface.co/datasets/BerkIGuler/PilotWiMAEDataset/). This dataset was used in the [PilotWiMAE paper](https://arxiv.org/abs/2605.22856).

## Generating channels from a config

After you define a YAML config (see examples under `config/` and `docs/main.pdf`), run:

```bash
python scripts/run.py --config path/to/your_config.yaml
```

Run from the repository root (or ensure Python can resolve the `src` package as in `scripts/run.py`). The script loads the config, streams channel outputs per transmit sector, and writes timestamped artifacts under `output/` (see `docs/main.pdf` for layout and metadata).

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

Read **`docs/main.pdf`** for:

- Full **YAML configuration reference** (every parameter and allowed values).
- How **scene generation** (Geo2SigMap / `scenegen`) ties into CSIGen.
- How **channel generation** runs end-to-end and what assumptions the pipeline makes (measurement surface, BS placement, sampling, LoS labeling, outputs, etc.).

Rebuild the PDF manual from source if needed:

```bash
./scripts/compile_docs.sh
```

## Purpose and scope

CSIGen is a **pipeline for producing large-scale wireless channel datasets** aimed at **training and evaluating wireless physical-layer AI models**. It builds on NVIDIA’s **Sionna** ray-tracing primitives but exposes a **single config file** to drive scene setup, sampling, solvers, and CSI export so you can scale generation without touching core code for each experiment. Configuration is documented in depth in **`docs/main.pdf`**.

This project is released **publicly** to help accelerate **wireless AI research**. **Feedback, issues, and fixes are welcome.**

## Repository layout

```
├── src/              # Library: scene setup, radio map / path solvers, CFR, config validation
├── config/           # YAML configs; examples under examples/
│                     #   (eval/ and pretrain/ city configs)
├── scenes/           # Example scenes: <city>_1/scene.xml (+ meshes referenced there)
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

## License

This software is licensed under the **MIT License**—see [`LICENSE`](LICENSE). The software is provided **“as is”**, without warranty of any kind. See the license file for the full disclaimer and terms.
