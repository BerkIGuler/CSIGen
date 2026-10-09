"""
Train/test splits of a generated run.

A run stores one file per TX, and each row of a file is the channel between
that TX and one user. A split assigns every channel (TX, user) to train, to
test or to neither:

- A user split (``users`` or ``area``) gives each user a side; all channels of
  a user follow it.
- A TX split (``sector`` or ``bs``) gives each TX a side; all channels of a TX
  follow it.
- With both, or with ``disjoint_users`` on a TX split (a user then takes the
  side of the TX it was sampled for), a channel is kept only when its TX and
  its user are on the same side.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

logger = logging.getLogger(__name__)

USER_SPLITS = ("users", "area")
TX_SPLITS = ("sector", "bs")
TRAIN, TEST, DROPPED = 0, 1, -1
SIDE_NAMES = {TRAIN: "train", TEST: "test"}

# Per-row metadata fields, sliced together with the CFR rows
ROW_FIELDS = ("rx_positions", "rx_names", "los_binary", "rx_serving_tx")

# What each method holds out, for split.yaml
TEST_LABELS = {
    "users": "Random users are test; all channels of a user are on the same side.",
    "area": "Users in randomly chosen square blocks of the scene are test; all channels of a user "
            "are on the same side.",
    "sector": "Some sectors (TXs) are test; the users can appear on both sides, with different sectors.",
    "sector_disjoint": "Some sectors (TXs) are test, and each user is on the side of the sector it was "
                       "sampled for; channels between a user and a sector of the other side are dropped.",
    "bs": "Some base stations, with all their sectors, are test; the users can appear on both sides, "
          "with different base stations.",
    "bs_disjoint": "Some base stations, with all their sectors, are test, and each user is on the side of the "
                   "base station it was sampled for; channels between a user and a base station of the other "
                   "side are dropped.",
    "sector+users": "Test sectors with random test users; channels between a test and a train item are dropped.",
    "sector+area": "Test sectors with users in test blocks; channels between a test and a train item are dropped.",
    "bs+users": "Test base stations with random test users; channels between a test and a train item are dropped.",
    "bs+area": "Test base stations with users in test blocks; channels between a test and a train item "
               "are dropped.",
}


def _tx_files(run_dir: Path) -> List[Path]:
    files = sorted(run_dir.glob("channel_tx_*.npz"), key=lambda f: int(f.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"No channel_tx_*.npz files in {run_dir}")
    return files


def _serving_tx(meta: Dict, rx_names: Sequence[str]) -> np.ndarray:
    """TX each user was sampled for; derived from the user numbering for runs without rx_serving_tx."""
    if "rx_serving_tx" in meta:
        return np.asarray(meta["rx_serving_tx"], dtype=int)
    if "users_per_tx" in meta:
        counts = np.asarray(meta["users_per_tx"], dtype=int)
    else:  # runs before 0.3.0: the same number of users for every TX
        counts = np.full(int(meta["num_txs"]), int(meta["num_users_per_tx"]))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    ids = np.array([int(n.split("_")[1]) for n in rx_names], dtype=int)
    return np.searchsorted(offsets, ids, side="right") - 1


def read_run_index(run_dir: Path) -> Dict:
    """
    Read the per-row metadata of every TX file of a run (not the CFRs).

    Returns a dict with the TX indices, the number of sectors, and for each
    user its position and serving TX.
    """
    run_dir = Path(run_dir)
    txs, rows, user_pos, user_serving = [], {}, {}, {}
    num_sectors = None
    for f in _tx_files(run_dir):
        with np.load(f, allow_pickle=True) as z:
            meta = z["metadata"].item()
        tx = int(meta["tx_idx"])
        names = [str(n) for n in meta["rx_names"]]
        pos = np.asarray(meta["rx_positions"]).reshape(len(names), 3)
        serving = _serving_tx(meta, names)
        for n, p, s in zip(names, pos, serving):
            user_pos.setdefault(n, p)
            user_serving.setdefault(n, int(s))
        txs.append(tx)
        rows[tx] = names
        num_sectors = int(meta["num_sectors"])
    users = sorted(user_pos, key=lambda n: int(n.split("_")[1]))
    return dict(
        txs=sorted(txs),
        rows=rows,
        num_sectors=num_sectors,
        users=users,
        positions=np.array([user_pos[n] for n in users], dtype=float).reshape(-1, 3),
        serving_tx=np.array([user_serving[n] for n in users], dtype=int),
    )


def _num_test(n: int, test_ratio: float) -> int:
    """Number of test items out of n: the rounded ratio, at least 1 and at most n - 1."""
    if n < 2:
        raise ValueError(f"Cannot split {n} item(s) into train and test")
    return int(min(max(round(test_ratio * n), 1), n - 1))


def position_groups(positions: np.ndarray) -> np.ndarray:
    """Group id of each user; users at the same position (to the millimeter) share a group."""
    _, group = np.unique(np.round(positions, 3), axis=0, return_inverse=True)
    return group.reshape(-1)


def split_users_random(positions: np.ndarray, test_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """
    Side of each user: a random subset of round(test_ratio * count) positions is test.

    Users at the same position (possible in runs before CSIGen 0.3.0) are
    drawn together, so they are always on the same side.
    """
    group = position_groups(positions)
    num_groups = int(group.max()) + 1
    test_groups = rng.choice(num_groups, size=_num_test(num_groups, test_ratio), replace=False)
    return np.where(np.isin(group, test_groups), TEST, TRAIN)


def drop_split_positions(side: np.ndarray, positions: np.ndarray) -> int:
    """
    Drop users whose position is shared with a user on the other side.

    Users at the same position have nearly the same channels, so they must
    not end up on different sides. ``side`` is changed in place; returns the
    number of dropped users.
    """
    group = position_groups(positions)
    num_dropped = 0
    for g in np.unique(group):
        members = np.flatnonzero(group == g)
        if len(members) > 1 and {TRAIN, TEST} <= set(side[members].tolist()):
            num_dropped += int((side[members] != DROPPED).sum())
            side[members] = DROPPED
    return num_dropped


def split_users_area(
    positions: np.ndarray,
    test_ratio: float,
    block_size: float,
    guard: float,
    rng: np.random.Generator,
) -> Dict:
    """
    Side of each user by area.

    The bounding box of the user positions is tiled with square blocks of
    ``block_size`` meters. Blocks are taken in random order and assigned to
    test until the test blocks hold at least ``test_ratio`` of the users
    (all blocks but one at most). Block [i, j] covers x in
    [x0 + i * block_size, x0 + (i + 1) * block_size) and likewise for y, where
    (x0, y0) is ``block_origin``, the smallest user x and y.
    Train users within ``guard`` meters (horizontal distance) of a test user
    are dropped, so no train user lies next to a test user.
    """
    xy = positions[:, :2]
    origin = xy.min(axis=0)
    cell = np.floor((xy - origin) / block_size).astype(int)
    blocks, block_of_user = np.unique(cell, axis=0, return_inverse=True)
    block_of_user = block_of_user.reshape(-1)
    if len(blocks) < 2:
        raise ValueError(f"block_size {block_size} m gives a single block; use smaller blocks")
    counts = np.bincount(block_of_user, minlength=len(blocks))
    order = rng.permutation(len(blocks))
    target = test_ratio * len(xy)
    test_blocks, num = [], 0
    for b in order:
        if num >= target:
            break
        test_blocks.append(int(b))
        num += counts[b]
    if len(test_blocks) == len(blocks):
        test_blocks.pop()
    side = np.where(np.isin(block_of_user, test_blocks), TEST, TRAIN)

    num_guarded = 0
    if guard > 0:
        test_xy = xy[side == TEST]
        train_idx = np.flatnonzero(side == TRAIN)
        for chunk in np.array_split(train_idx, max(1, len(train_idx) // 1000)):
            d2 = ((xy[chunk, None, :] - test_xy[None, :, :]) ** 2).sum(axis=2)
            near = chunk[(d2 < guard ** 2).any(axis=1)]
            side[near] = DROPPED
            num_guarded += len(near)
    return dict(
        side=side,
        test_blocks=sorted(blocks[test_blocks].tolist()),
        block_origin=[float(v) for v in origin],
        num_blocks=int(len(blocks)),
        num_guarded_users=int(num_guarded),
    )


def split_txs(
    txs: Sequence[int],
    by: str,
    num_sectors: int,
    test_ratio: float,
    rng: np.random.Generator,
    test_ids: Optional[Sequence[int]] = None,
    nonempty_txs: Optional[Sequence[int]] = None,
) -> Dict:
    """
    Side of each TX, by sector (``by="sector"``) or by base station (``by="bs"``).

    ``test_ids`` lists the test sectors (TX indices) or base stations;
    otherwise round(test_ratio * count) of them are drawn at random among
    those with channels (``nonempty_txs``), so that a sector without users
    is never drawn as test. Base station b holds the TXs b * num_sectors to
    (b + 1) * num_sectors - 1.
    """
    txs = np.asarray(txs)
    groups = txs if by == "sector" else txs // num_sectors
    ids = np.unique(groups)
    if test_ids is None:
        nonempty = ids if nonempty_txs is None else np.unique(
            groups[np.isin(txs, list(nonempty_txs))])
        test_ids = np.sort(rng.choice(nonempty, size=_num_test(len(nonempty), test_ratio), replace=False))
    test_ids = np.asarray(sorted(int(i) for i in test_ids))
    unknown = set(test_ids.tolist()) - set(ids.tolist())
    if unknown:
        raise ValueError(f"Unknown test {by} ids: {sorted(unknown)}")
    if len(test_ids) == len(ids):
        raise ValueError(f"All {by}s are test; leave at least one for train")
    side = np.where(np.isin(groups, test_ids), TEST, TRAIN)
    return dict(side={int(t): int(s) for t, s in zip(txs, side)}, test_ids=test_ids.tolist())


def channel_side(tx_side: Optional[int], user_side: Optional[int]) -> int:
    """Side of a channel from the sides of its TX and its user (None: not split on that axis)."""
    if tx_side is None:
        return user_side
    if user_side is None:
        return tx_side
    return tx_side if tx_side == user_side else DROPPED


def git_commit(repo: Path) -> Optional[str]:
    """Git commit of the CSIGen repo, with '-dirty' for uncommitted changes, or None."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _rename_source_counts(meta: Dict) -> None:
    """
    Rename the user counts of the source run, which do not hold for a split run.

    users_per_tx and total_users (num_users_per_tx in runs before 0.3.0)
    describe the users sampled in the source run, and the UE_k numbering
    they imply does not hold for the users of one side.
    """
    for key in ("users_per_tx", "num_users_per_tx", "total_users"):
        if key in meta:
            meta[f"source_{key}"] = meta.pop(key)


def write_split(run_dir: Path, out_dir: Path, tx_side: Dict[int, int], user_side: Dict[str, int],
                split_info: Dict) -> Dict:
    """
    Write the train and test runs and their split.yaml.

    Each side gets every TX file of the run, restricted to the channels of
    that side (possibly none), with ``source_rows`` holding the row indices
    of the kept channels in the source file.
    """
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    dirs = {s: out_dir / name for s, name in SIDE_NAMES.items()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "metadata.yaml") as f:
        run_meta = yaml.load(f, Loader=yaml.FullLoader)

    channels = {s: {} for s in SIDE_NAMES}
    users = {s: set() for s in SIDE_NAMES}
    shapes = {s: [] for s in SIDE_NAMES}
    num_dropped = 0
    for f in _tx_files(run_dir):
        with np.load(f, allow_pickle=True) as z:
            h = z["h"]
            meta = z["metadata"].item()
        tx = int(meta["tx_idx"])
        names = [str(n) for n in meta["rx_names"]]
        sides = np.array([channel_side(tx_side.get(tx), user_side.get(n)) for n in names], dtype=int)
        num_dropped += int((sides == DROPPED).sum())
        for s, d in dirs.items():
            rows = np.flatnonzero(sides == s)
            m = dict(meta)
            for key in ROW_FIELDS:
                if key in m:
                    m[key] = np.asarray(m[key])[rows]
            m["rx_names"] = [names[i] for i in rows]
            m["source_rows"] = rows.astype(np.int64)
            m["num_valid_channels"] = int(len(rows))
            m["cfr_shape"] = (len(rows),) + tuple(h.shape[1:])
            m["split_side"] = SIDE_NAMES[s]
            _rename_source_counts(m)
            np.savez_compressed(d / f.name, h=h[rows], shape=m["cfr_shape"], dtype=str(h.dtype), metadata=m)
            channels[s][tx] = int(len(rows))
            users[s].update(m["rx_names"])
            shapes[s].append(list(m["cfr_shape"]))
        logger.info("%s: %s train, %s test, %s dropped", f.name,
                    channels[TRAIN][tx], channels[TEST][tx], int((sides == DROPPED).sum()))

    summary = dict(
        **split_info,
        num_channels={SIDE_NAMES[s]: int(sum(c.values())) for s, c in channels.items()},
        num_channels_dropped=num_dropped,
        num_users={SIDE_NAMES[s]: len(u) for s, u in users.items()},
        num_users_in_both=len(users[TRAIN] & users[TEST]),
        channels_per_tx={SIDE_NAMES[s]: [c[t] for t in sorted(c)] for s, c in channels.items()},
    )
    for s, d in dirs.items():
        meta = dict(run_meta, cfr_per_tx_shapes=shapes[s], split_side=SIDE_NAMES[s], num_users=len(users[s]))
        _rename_source_counts(meta)
        with open(d / "metadata.yaml", "w") as f:
            yaml.dump(meta, f, default_flow_style=False)
        with open(d / "split.yaml", "w") as f:
            yaml.dump(dict(summary, side=SIDE_NAMES[s]), f, default_flow_style=False, sort_keys=False)
    return summary
