"""Synthetic path data and runs in the CSIGen file format, written without Sionna RT."""

import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from csigen.path_data import cfr_from_paths

REPO = Path(__file__).resolve().parents[1]
CFR = {"carrier_frequency": 3.5e9, "num_subcarriers": 8, "num_ofdm_symbols": 3, "subcarrier_spacing": 30e3,
       "normalize_delays": False, "normalize": False}
NUM_RX_ANT, NUM_TX_ANT, MAX_DEPTH = 2, 3, 2
# Planar arrays of one row, with the keys of the array layouts in the metadata of real files
TX_ARRAY = {"num_rows": 1, "num_cols": NUM_TX_ANT, "num_polarizations": 1, "num_ant": NUM_TX_ANT,
            "vertical_spacing": 0.5, "horizontal_spacing": 0.5, "pattern": "iso", "polarization": "V"}
RX_ARRAY = {"num_rows": 1, "num_cols": NUM_RX_ANT, "num_polarizations": 1, "num_ant": NUM_RX_ANT,
            "vertical_spacing": 0.5, "horizontal_spacing": 0.5, "pattern": "iso", "polarization": "V"}


def command(name):
    """A CSIGen command of the environment that runs the tests, else the one on the PATH."""
    found = shutil.which(name, path=str(Path(sys.executable).parent)) or shutil.which(name)
    assert found, f"the command {name} is not installed (pip install -e .)"
    return found


def element_phases(layout, theta, phi):
    """
    Plane-wave phase of each element of a planar array without rotation, for the direction (theta, phi) of each
    path, [P, num_rows * num_cols]. The array lies in the y-z plane, and the vertical index runs fastest.
    """
    rows, cols = layout["num_rows"], layout["num_cols"]
    r, c = np.tile(np.arange(rows), cols), np.repeat(np.arange(cols), rows)
    y = layout["horizontal_spacing"] * (c - (cols - 1) / 2)
    z = -layout["vertical_spacing"] * (r - (rows - 1) / 2)
    theta, phi = np.asarray(theta, np.float64)[:, None], np.asarray(phi, np.float64)[:, None]
    return 2 * np.pi * (np.sin(theta) * np.sin(phi) * y + np.cos(theta) * z)


def random_paths(counts, los=None, seed=0, arrays=False):
    """
    Path data of len(counts) channels with counts[i] paths each, sorted by power as CSIGen saves them. The first
    path of channel i has no interaction when los[i] is true, and los defaults to true for every channel. The other
    paths bounce once or twice. With arrays, paths_a is that of the synthetic arrays TX_ARRAY and RX_ARRAY without
    rotation, a centre coefficient times a plane-wave phase per element. Otherwise each antenna pair has a random
    coefficient.
    """
    rng = np.random.default_rng(seed)
    counts = np.asarray(counts)
    los = np.ones(len(counts), dtype=bool) if los is None else np.asarray(los)
    p = int(counts.sum())
    a = ((rng.normal(size=(p, NUM_RX_ANT, NUM_TX_ANT)) + 1j * rng.normal(size=(p, NUM_RX_ANT, NUM_TX_ANT)))
         * 1e-5).astype(np.complex64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    angles = {key: rng.uniform(lo, hi, p).astype(np.float32) for key, lo, hi in (
        ("paths_theta_t", 0, np.pi), ("paths_phi_t", -np.pi, np.pi),
        ("paths_theta_r", 0, np.pi), ("paths_phi_r", -np.pi, np.pi))}
    if arrays:
        ph = (element_phases(RX_ARRAY, angles["paths_theta_r"], angles["paths_phi_r"])[:, :, None]
              + element_phases(TX_ARRAY, angles["paths_theta_t"], angles["paths_phi_t"])[:, None, :])
        a = (a[:, :1, :1].astype(np.complex128) * np.exp(1j * ph)).astype(np.complex64)
    power = np.mean(np.abs(a) ** 2, axis=(1, 2))
    order = np.concatenate([s + np.argsort(-power[s:e]) for s, e in zip(offsets[:-1], offsets[1:])]).astype(int)
    inter = rng.choice([1, 2, 4, 8], size=(p, MAX_DEPTH)).astype(np.uint8)
    inter[:, 1][rng.random(p) < 0.5] = 0
    inter[offsets[:-1][(counts > 0) & los]] = 0
    return {"paths_offsets": offsets,
            "paths_a": a[order],
            "paths_tau": rng.uniform(1e-7, 2e-6, p).astype(np.float32),
            "paths_doppler": rng.uniform(-500, 500, p).astype(np.float32),
            **{key: v[order] for key, v in angles.items()},
            "paths_interactions": inter}


def serving_map(num_txs, users_per_tx):
    """A serving TX for each user that differs from the block numbering (the blocks in reverse order)."""
    return (num_txs - 1 - np.arange(num_txs * users_per_tx) // users_per_tx).astype(np.int32)


def write_run(run, num_bs=2, num_sectors=2, users_per_tx=3, layout="current", seed=0):
    """
    A run of num_bs base stations with num_sectors sectors each. Each sector samples users_per_tx users, numbered
    in blocks as CSIGen numbers them (UE_0 to UE_2 for TX 0, and so on). Each sector file holds the channels of
    the users that it serves and of a random part of the other users, in a random order. Thus the files differ in
    their rows and in the order of the users, and a user is found only by its name. The paths are those of the
    synthetic arrays TX_ARRAY and RX_ARRAY.

    layout sets how a file gives the serving TX of each user. "current" writes rx_serving_tx (serving_map, which
    differs from the blocks). "users_per_tx" leaves it out, as some runs of CSIGen 0.3.0 and later do. "old" writes
    only num_txs and num_users_per_tx, as runs before CSIGen 0.3.0 do.
    """
    rng = np.random.default_rng(seed)
    run.mkdir(parents=True)
    num_txs = num_bs * num_sectors
    num_users = num_txs * users_per_tx
    names = [f"UE_{u}" for u in range(num_users)]
    pos = np.c_[rng.uniform(0, 300, (num_users, 2)), np.full(num_users, 1.5)]
    blocks = np.arange(num_users) // users_per_tx
    serving = serving_map(num_txs, users_per_tx) if layout == "current" else blocks
    for tx in range(num_txs):
        ids = rng.permutation(np.flatnonzero((serving == tx) | (rng.random(num_users) < 0.5)))
        n = len(ids)
        counts = rng.integers(1, 4, n)
        los = rng.random(n) < 0.5
        paths = random_paths(counts, los, seed=seed + tx, arrays=True)
        h = cfr_from_paths(paths, **CFR)
        meta = {"tx_idx": tx, "num_txs": num_txs, "num_sectors": num_sectors, "rx_names": [names[i] for i in ids],
                "rx_positions": pos[ids], "los_binary": los.astype(int), "rx_velocities": rng.normal(size=(n, 3)),
                "rx_orientations": np.zeros((n, 3)), "tx_orientation": np.zeros(3), "num_valid_channels": n,
                "cfr_saved": True, "cfr_shape": h.shape, "cfr_dtype": str(h.dtype), "cfr_params": CFR,
                "cfr_source": "cfr_from_paths", "paths_saved": True,
                "tx_array": TX_ARRAY, "rx_array": RX_ARRAY,
                "path_stats": {"num_paths": counts.astype(np.int32)}}
        if layout == "old":
            meta["num_users_per_tx"] = users_per_tx
        else:
            meta.update(users_per_tx=[users_per_tx] * num_txs, total_users=num_users)
        if layout == "current":
            meta["rx_serving_tx"] = serving[ids].astype(np.int32)
        np.savez_compressed(run / f"channel_tx_{tx}.npz", h=h, shape=h.shape, dtype=str(h.dtype),
                            metadata=np.array(meta, dtype=object), **paths)
    with open(run / "metadata.yaml", "w") as f:
        yaml.dump({"scene_name": "toy_1", "run_timestamp": "20260101_000000", "csigen_version": "0.4.2",
                   "num_sectors": num_sectors, "total_users": num_users}, f)
    return run
