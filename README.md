# CSIGen

Config-driven wireless channel dataset generation built on **Sionna RT**. This README is short on purpose. Use the sections below in order—or jump to **`docs/main.pdf`** for the full manual.

## Generating channels from a config

After you define a YAML config (see examples under `config/` and `docs/main.pdf`), run:

```bash
python scripts/run.py --config path/to/your_config.yaml
```

Run from the repository root (or ensure Python can resolve the `src` package as in `scripts/run.py`). The script loads the config, streams channel outputs per transmit sector, and writes timestamped artifacts under `output/` (see `docs/main.pdf` for layout and metadata).

## Requirements

| Component | Version (reference) |
|-----------|---------------------|
| **Python** | **3.12.x** |
| **Sionna** | **1.2.1**  |

Install dependencies into your environment:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins packages as resolved in our system. Other platforms or CUDA stacks may need adjusted TensorFlow/Mitsuba wheels. Follow [Sionna](https://github.com/NVlabs/sionna) install guidance if `pip install` fails.

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
├── config/           # YAML templates; dataset presets under pilotwimae_dataset_configs/
│                     #   (eval/ and pretrain/ city configs)
├── scenes/           # Example scenes: <city>_1/scene.xml (+ meshes referenced there)
├── scripts/          # run.py, compile_docs.sh, and other CLI helpers
├── docs/             # main.tex and build script for the PDF manual (main.pdf)
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
