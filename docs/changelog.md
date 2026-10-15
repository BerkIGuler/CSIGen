# Changelog

All notable changes to CSIGen are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.3.5] - 2026-10-14

### Added

- **Base stations on masts.** The new optional keys `bs_positions` (a list of `[x, y]` positions in meters, in the scene's frame) and `bs_mast_height` (meters) place one base station at each position, with its antennas `bs_mast_height` above the terrain there: the height of the terrain triangle that contains the position, interpolated within the triangle, so the masts follow hills. They replace the rooftop placement, which puts base stations on the highest buildings and, in suburban or rural scenes, on houses; `num_deployment_buildings` and `antenna_height_offset`, otherwise required, can then be left out. Give both keys or neither. The sectors, downtilt and power are as before, and the positions are recorded as the TX positions in the metadata. A position outside the clipped terrain is an error, as is an antenna below the top of a building whose bounding box contains its position; an antenna above such a building logs a warning. Without the keys, the base stations are placed as before.

## [0.3.4] - 2026-10-10

### Changed

- **Path solver batching on by default.** `path_solver_rx_batch_size` now defaults to 50, and `path_solver_spec_table_size` to 400,000,000 with a synthetic array. Before, a config without these keys solved all receivers of a TX at once with Sionna's small specular-chain table, which silently loses some NLoS paths; such a config now gets the same paths as the example configs, which already set both values. Set a key to `null` for the old behaviour. Without a synthetic array, the table keeps Sionna's default, because every TX antenna is a source and a larger table would exceed Sionna's 32-bit indices; with a `cfr_out_type` other than `numpy`, batching is not possible there, and the receivers are solved at once with a warning.
- **Diffuse reflection off in the example configs.** `radio_map_diffuse_reflection` and `path_solver_diffuse_reflection` are now `false`. The materials of the included scenes have a scattering coefficient of 0, so the flags had no effect: with the flags on or off, the radio map samples the same users and the path solver finds the same paths, up to the usual run-to-run variation. Diffuse scattering needs scattering coefficients above 0.

## [0.3.3] - 2026-10-10

### Added

- **Path data.** With a synthetic array, each `channel_tx_<i>.npz` now also holds the paths of every channel: the coefficient per antenna pair (`paths_a`), the delay, the Doppler shift, the angles of departure and arrival, and the interaction type of each bounce, sorted by power, with `paths_offsets` marking the paths of each channel. All valid paths are kept. Their size grows with the number of paths: with a few paths per channel they take a few percent of the size of the CFR, but with many paths, for example from diffuse scattering, they can exceed it. Set `save_paths: false` to leave them out.
- **Path statistics and mobility.** The per-TX metadata adds `path_stats`, per-channel arrays computed from all paths: number of paths, path gain, TX-RX distance, first and mean delay, delay spread, K-factor, azimuth and zenith spreads of departure and arrival, the delay, angles and Doppler shift of the strongest path and its share of the power, the largest Doppler shift, the number of interactions of the strongest path and the largest over the paths, and the share of the power in the LoS path and in paths with a specular reflection, a diffuse reflection, a refraction or a diffraction. It also adds `tx_orientation`, and `rx_velocities` and `rx_orientations`, one row per channel.
- **CFR from the paths.** `load_cfr(file)` rebuilds the saved CFR of a file exactly from its paths. With other arguments, it computes the CFR without new ray tracing for another numerology or normalization (subcarriers, subcarrier spacing, OFDM symbols, symbol duration, for example with a cyclic prefix), for other planar arrays (`tx_array` and `rx_array` change the rows, columns and spacing; the element pattern, polarization and orientation stay those of the run), or for other RX velocities (`rx_velocities`, one velocity for all RXs or one per channel, in m/s; the Doppler shifts change, the paths stay those of the run), or from only some of the paths (`max_paths` keeps the strongest paths, `power_range_db` the paths within this many dB of the strongest, `max_bounces` the paths with at most this many interactions, 0 for LoS only, and `exclude_interactions` drops the paths with, for example, a diffraction). With `save_cfr: false`, the files hold only the paths. The split script also slices the paths and statistics.

### Changed

- **CFR computation.** With a synthetic array, CSIGen now computes the CFR itself from the saved paths, in float64, instead of calling Sionna's `Paths.cfr`, so the saved CFR is exactly reproducible from the saved paths. The formula is the same, but Sionna computes the carrier phase in float32, which is accurate to about 0.01 rad at 28 GHz, so new CFRs differ from those of earlier versions by about 1% at 28 GHz and 0.1% at 3.5 GHz, with no systematic change. Neither is more accurate, since the delays are float32 in both. The per-TX metadata records the CFR parameters in `cfr_params`, how the CFR was computed in `cfr_source`, and whether `h` was saved in `cfr_saved`. Without a synthetic array, Sionna's `Paths.cfr` is used as before, and paths cannot be saved.

## [0.3.2] - 2026-10-09

### Added

- **Train/test splits.** `scripts/split_dataset.py` splits a run into a train run and a test run in the usual format, so that the two share no users, no sectors, no base stations, or neither users nor TXs. `--by` selects random users (`users`), square blocks of the scene with an optional guard distance (`area`), sectors (`sector`), base stations (`bs`), or a sector or base-station split combined with a user split (for example `bs+area`), with `--test-ratio` or explicit `--test-ids`, and `--seed`. Each side gets a `split.yaml` that records the method, its parameters, the chosen test items, the source run, the CSIGen version and git commit, and the counts, and each file records the source row of every kept channel in `source_rows`. Users at the same position are never split across the two sides. In a split run, the user counts of the source run are renamed to `source_users_per_tx` and `source_total_users`.

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
