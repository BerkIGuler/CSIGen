# Changelog

All notable changes to CSIGen are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-10-09

### Fixed

- **Duplicate and missing users.** Sionna RT samples each TX's users from its valid radio-map cells with replacement, so a TX with few valid cells placed many users at the same cell center, and a TX with no valid cell got NaN positions, which made the edge filter remove every user. Each TX now gets up to `num_user_samples_per_tx` users in distinct cells, so the number of users per TX can differ, and a TX without valid cells gets none. Each cell gets a random key from `user_sample_seed`, and each TX takes its valid cells with the lowest keys. The radio map is not bit-identical between GPU runs, so a few cells near the thresholds can change validity between runs with the same config; with the keys, such a change moves at most one user instead of most of the draw.

### Changed

- **Metadata.** `num_users_per_tx` is replaced by `users_per_tx` (the user count of each TX), and the per-TX metadata adds `rx_serving_tx`, the TX each saved user was sampled for.

## [0.2.3] - 2026-10-09

### Fixed

- **Paths lost with many receivers.** Sionna RT finds reflected paths by remembering each new chain of reflecting surfaces per receiver in a hash table of fixed size. When many receivers share one path solve, the table fills up and new chains are discarded as duplicates without a warning, so receivers, mostly NLoS ones, lose some paths or all of them. Two new settings control this:
  - `path_solver_rx_batch_size` solves each TX's receivers in batches of this size, so fewer receivers compete for the table. Smaller batches are more accurate but slower.
  - `path_solver_spec_table_size` enlarges the table (8 bytes of GPU memory per entry), so larger batches stay accurate.

- **Receivers without paths in a solve.** A solve in which no receiver has a path now marks each receiver as having no path instead of failing.

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
