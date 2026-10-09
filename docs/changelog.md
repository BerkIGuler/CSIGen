# Changelog

All notable changes to CSIGen are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-10-09

### Added

- **Path buffer check.** Sionna RT keeps at most `path_solver_max_num_paths_per_src` candidate paths per source in one solve and discards the rest without a warning, which removes mostly NLoS paths. CSIGen now measures how full this buffer gets in each solve, logs a warning when it is full, and saves the largest fill of each TX as `path_buffer_fill` in the per-TX metadata (a fraction of the cap; at 1.0, paths were discarded). If it happens, lower `path_solver_rx_batch_size` or raise `path_solver_max_num_paths_per_src`.

### Changed

- **Path solver API.** `solve_paths_per_tx` is replaced by `iter_paths_per_tx`, which solves each TX's receivers in batches (`rx_batch_size`, default 50) and yields one batch at a time, so it loses no more paths than `generate_channels`. The old function solved all receivers of a TX at once, which loses paths, and kept every `Paths` object; each holds about 2 GB of GPU memory with 10^7 paths per source. `iter_paths_for_receivers` does the same for one TX, and the statistics helpers accept lists of batches.

### Fixed

- **Silent index overflow.** Sionna RT indexes its specular-chain table and path buffer with 32-bit integers, so `max(path_solver_spec_table_size, path_solver_max_num_paths_per_src)` times the number of sources must stay below 2^32. There is one source per TX antenna without a synthetic array, so, for example, the 4e8 table with a 32-antenna array would overflow and discard paths silently. The config validator now rejects such settings.
- **Table size between runs.** `path_solver_spec_table_size` changes a setting of Sionna RT that is shared by the whole process. A run without the option now restores Sionna's default instead of keeping the size of an earlier run.

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
