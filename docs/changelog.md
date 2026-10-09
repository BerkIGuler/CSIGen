# Changelog

All notable changes to CSIGen are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.2.2] - 2026-10-08

### Added

- **Array layout in the metadata.** The metadata records the CFR axes and, for each array, its layout and antenna index order.

## [0.2.1] - 2026-10-08

### Added

- **Asphalt ground.** `override_ground_material` accepts `asphalt_concrete` (ITU-R P.2040-4, 1-40 GHz) as well as `concrete`.

### Changed

- **Example configs.** The configs in `config/examples/` use `asphalt_concrete` ground.

## [0.2.0] - 2026-10-08

### Changed

- **Sionna RT 2.2.0.** CSIGen runs on Sionna RT 2.2.0 instead of 1.2.1, and `requirements.txt` lists only the packages CSIGen imports.

## [0.1.1] - 2026-10-08

### Fixed

- **Mechanical tilt.** A positive `mechanical_tilt` now tilts the antennas down, as documented; earlier versions tilted them up.

### Added

- **Version in the metadata.** `scripts/run.py` writes the CSIGen version to `metadata.yaml` as `csigen_version`.

## [0.1.0] - 2026-05-17

First versioned release: config-driven channel generation with Sionna RT 1.2.1.
