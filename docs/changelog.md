# Changelog

All notable changes to CSIGen are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-10-08

### Added

- **Asphalt ground.** `override_ground_material` accepts `asphalt_concrete` (ITU-R P.2040-4, 1-40 GHz) as well as `concrete`.

### Changed

- **Ground material of the dataset configs.** The configs in `config/pilotwimae_dataset_configs/` now accept `asphalt_concrete`.

## [0.2.0] - 2026-10-08

### Changed

- **Sionna RT 2.2.0.** CSIGen now runs on Sionna RT 2.2.0 (Mitsuba 3.9.1, Dr.Jit 1.5.0) instead of 1.2.1. The code needed no changes; `requirements.txt` now lists only Sionna RT and the packages CSIGen imports, and drops TensorFlow and the full Sionna package.
- **NVIDIA driver.** We tested Sionna RT 2.2.0 on the GPU with NVIDIA driver 580, and it works.

## [0.1.1] - 2026-10-08

### Fixed

- **Sign of the mechanical tilt.** `add_base_station` (`src/base_station.py`) passed `-mechanical_tilt` as the second Sionna orientation angle (beta). In Sionna RT, a positive beta tilts the boresight below the horizon, so a positive `mechanical_tilt` gave an **uptilt** instead of the documented downtilt. The value is now passed with a positive sign, so `mechanical_tilt: 10` points the boresight 10 degrees below the horizon.

  Data generated with 0.1.0 or earlier used an uptilt of `mechanical_tilt` degrees. This includes the PilotWiMAE dataset, which was generated with the configs in `config/pilotwimae_dataset_configs/` (`mechanical_tilt: 10.0`). With the tr38901 pattern (65 degree vertical beamwidth), the element gain toward a user 10 degrees below the horizon is about 1 dB lower than with the intended downtilt. To reproduce that data, use version 0.1.0.

### Added

- `src.__version__`, and a `csigen_version` field in the `metadata.yaml` written by `scripts/run.py`, so that each dataset records the version that produced it.

## [0.1.0] - 2026-05-17

First versioned release. Config-driven channel generation with Sionna RT 1.2.1: scene setup from Geo2SigMap scenes, multi-sector base stations, radio-map-based user sampling, mobility presets, per-sector path solving, OFDM CFR export with per-sector metadata (positions, receiver names, LoS flags), and the configs that produced the PilotWiMAE dataset.

### Known issues

- `mechanical_tilt` gives an uptilt instead of a downtilt (fixed in 0.1.1).
