import subprocess
import sys

import numpy as np
import pytest
import yaml

from helpers import REPO, serving_map, write_run
from src.path_data import load_cfr
from src.split import (DROPPED, TEST, TRAIN, drop_split_positions, read_run_index, split_txs, split_users_area,
                       split_users_random, write_split)


def read_side(out, side):
    files = {}
    for f in sorted((out / side).glob("channel_tx_*.npz")):
        with np.load(f, allow_pickle=True) as z:
            files[f.name] = (z["h"], z["metadata"].item())
    with open(out / side / "split.yaml") as f:
        return files, yaml.safe_load(f)


def side_users(out):
    """The users of each side, as two sets."""
    return [{n for _, m in read_side(out, s)[0].values() for n in m["rx_names"]} for s in ("train", "test")]


def channels_per_tx(index, keep):
    """The number of rows of each sector file whose sector and user pass keep(tx, user)."""
    return [sum(keep(tx, u) for u in index["rows"][tx]) for tx in index["txs"]]


def check_rows(run_dir, out):
    """Every kept row equals its source row, and its paths rebuild its CFR."""
    for side in ("train", "test"):
        for name, (h, meta) in read_side(out, side)[0].items():
            with np.load(run_dir / name, allow_pickle=True) as z:
                src_h, src_meta = z["h"], z["metadata"].item()
            rows = meta["source_rows"]
            assert np.array_equal(h, src_h[rows])
            assert meta["rx_names"] == [src_meta["rx_names"][r] for r in rows]
            assert np.array_equal(meta["path_stats"]["num_paths"], src_meta["path_stats"]["num_paths"][rows])
            assert load_cfr(out / side / name).tobytes() == h.tobytes()


def test_user_split_is_disjoint(run_dir, tmp_path):
    index = read_run_index(run_dir)
    side = split_users_random(index["positions"], 0.25, np.random.default_rng(1))
    out = tmp_path / "split"
    summary = write_split(run_dir, out, {}, dict(zip(index["users"], side.tolist())), {"method": "users"})
    train, test = side_users(out)
    assert not train & test
    assert len(test) == 3 and summary["num_users"] == {"train": 9, "test": 3}
    sides = dict(zip(index["users"], side))
    for name, s in (("train", TRAIN), ("test", TEST)):
        assert summary["channels_per_tx"][name] == channels_per_tx(index, lambda tx, u: sides[u] == s)
    assert summary["num_channels_dropped"] == 0
    info = read_side(out, "test")[1]
    assert info["side"] == "test" and info["num_channels"] == summary["num_channels"]
    check_rows(run_dir, out)


def test_base_station_and_user_split(run_dir, tmp_path):
    index = read_run_index(run_dir)
    rng = np.random.default_rng(0)
    tx_side = split_txs(index["txs"], "bs", index["num_sectors"], 0.5, rng, test_ids=[1])["side"]
    assert tx_side == {0: TRAIN, 1: TRAIN, 2: TEST, 3: TEST}
    user_side = split_users_random(index["positions"], 0.25, rng)
    out = tmp_path / "split"
    summary = write_split(run_dir, out, tx_side, dict(zip(index["users"], user_side.tolist())), {"method": "bs+users"})
    # A channel is kept only when its sector and its user are on the same side
    sides = dict(zip(index["users"], user_side))
    for name, s in (("train", TRAIN), ("test", TEST)):
        assert summary["channels_per_tx"][name] == channels_per_tx(
            index, lambda tx, u: tx_side[tx] == s and sides[u] == s)
    kept = summary["num_channels"]["train"] + summary["num_channels"]["test"]
    assert 0 < summary["num_channels_dropped"] == sum(map(len, index["rows"].values())) - kept
    check_rows(run_dir, out)


@pytest.mark.parametrize("layout", ["current", "users_per_tx", "old"])
def test_serving_tx_of_each_user(tmp_path, layout):
    # Without rx_serving_tx, the serving TX follows from the user numbering
    index = read_run_index(write_run(tmp_path / "run", layout=layout))
    assert index["users"] == [f"UE_{u}" for u in range(12)]
    blocks = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    expected = serving_map(4, 3).tolist() if layout == "current" else blocks
    assert index["serving_tx"].tolist() == expected


def test_disjoint_base_station_split(run_dir, tmp_path):
    # Each user takes the side of the base station that serves it, and every user has a row in its serving sector
    index = read_run_index(run_dir)
    tx_side = split_txs(index["txs"], "bs", index["num_sectors"], 0.5, np.random.default_rng(0), test_ids=[0])["side"]
    user_side = {u: tx_side[int(s)] for u, s in zip(index["users"], index["serving_tx"])}
    out = tmp_path / "split"
    summary = write_split(run_dir, out, tx_side, user_side, {"method": "bs"})
    train, test = side_users(out)
    assert not train & test and summary["num_users_in_both"] == 0
    # serving_map serves UE_6 to UE_11 from TXs 1 and 0, which form base station 0
    assert test == {f"UE_{u}" for u in range(6, 12)} and train == {f"UE_{u}" for u in range(6)}
    assert summary["channels_per_tx"]["test"] == channels_per_tx(index, lambda tx, u: tx < 2 and u in test)
    assert summary["channels_per_tx"]["train"] == channels_per_tx(index, lambda tx, u: tx >= 2 and u in train)
    check_rows(run_dir, out)


def test_area_split_keeps_the_guard():
    rng = np.random.default_rng(0)
    pos = np.c_[rng.uniform(0, 300, (200, 2)), np.full(200, 1.5)]
    res = split_users_area(pos, 0.25, 100.0, 20.0, np.random.default_rng(1))
    side = res["side"]
    assert (side == TEST).sum() >= 50 and (side == TRAIN).any()
    assert res["num_guarded_users"] == int((side == DROPPED).sum()) > 0
    d = np.linalg.norm(pos[side == TRAIN, None, :2] - pos[None, side == TEST, :2], axis=2)
    assert d.min() >= 20.0
    # Each test block holds only test users and dropped users
    cell = np.floor((pos[:, :2] - res["block_origin"]) / 100.0).astype(int)
    in_test = np.array([list(c) in res["test_blocks"] for c in cell])
    assert np.all(side[in_test] == TEST) and not np.any(side[~in_test] == TEST)


def test_users_at_one_position_are_on_one_side():
    pos = np.array([[0, 0, 1.5], [0, 0, 1.5], [5, 5, 1.5], [9, 9, 1.5]])
    side = np.array([TRAIN, TEST, TEST, TRAIN])
    assert drop_split_positions(side, pos) == 2
    assert side.tolist() == [DROPPED, DROPPED, TEST, TRAIN]


def test_empty_sectors_are_never_test():
    for seed in range(20):
        res = split_txs([0, 1, 2, 3], "sector", 2, 0.5, np.random.default_rng(seed), nonempty_txs=[1, 3])
        assert set(res["test_ids"]) <= {1, 3}


@pytest.mark.parametrize("args, test_key", [
    (["--by", "sector", "--test-ratio", "0.25"], "test_sectors"),
    (["--by", "bs", "--disjoint-users"], "test_base_stations"),
    (["--by", "area", "--block-size", "100", "--guard", "10"], "test_blocks"),
])
def test_split_script(run_dir, tmp_path, args, test_key):
    out = tmp_path / "split"
    cmd = [sys.executable, str(REPO / "scripts" / "split_dataset.py"), "--run", str(run_dir), "--out", str(out),
           "--seed", "1", *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    info = read_side(out, "test")[1]
    assert info["method"] == args[1] and info[test_key]
    total = info["num_channels"]["train"] + info["num_channels"]["test"] + info["num_channels_dropped"]
    assert total == sum(map(len, read_run_index(run_dir)["rows"].values()))
    if args[1] != "sector":                         # the user is on one side only
        train, test = side_users(out)
        assert not train & test and info["num_users_in_both"] == 0
    check_rows(run_dir, out)
