"""
Split a generated run into a train run and a test run.

Methods (combine a user split and a TX split with --by, e.g. "sector+area"):
  users   random users; all channels of a user go to the same side
  area    users by area: square blocks of the scene, with an optional guard
  sector  N of the M sectors (TXs) are test
  bs      N of the base stations (all their sectors) are test

The output directory gets train/ and test/, each a run in the usual format
(channel_tx_*.npz + metadata.yaml) plus split.yaml, which records the method,
its parameters, the chosen sectors, base stations or blocks, and the counts.

Example:
  python scripts/split_dataset.py --run output/chicago_1/20261009_100551 \\
      --out output/chicago_1/20261009_100551_split_bs --by bs --test-ratio 0.2 --seed 1
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import yaml

from src import __version__
from src.split import (
    TEST_LABELS,
    USER_SPLITS,
    TX_SPLITS,
    read_run_index,
    split_users_random,
    drop_split_positions,
    split_users_area,
    split_txs,
    git_commit,
    write_split,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

METHODS = ("users", "area", "sector", "bs", "sector+users", "sector+area", "bs+users", "bs+area")


def parse_args():
    parser = argparse.ArgumentParser(description="Split a generated run into train and test runs")
    parser.add_argument("--run", type=Path, required=True, help="Run directory (channel_tx_*.npz + metadata.yaml)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory; train/ and test/ are created in it")
    parser.add_argument("--by", choices=METHODS, required=True, help="Split method")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of the users, sectors or base stations that are test (default 0.2); "
                             "a combined split applies it to both")
    parser.add_argument("--test-ids", type=int, nargs="+",
                        help="Test sectors (TX indices) or base stations, instead of drawing them")
    parser.add_argument("--disjoint-users", action="store_true",
                        help="sector/bs: a user takes the side of the TX it was sampled for, "
                             "and channels across sides are dropped")
    parser.add_argument("--block-size", type=float, default=50.0, help="area: block size in meters (default 50)")
    parser.add_argument("--guard", type=float, default=0.0,
                        help="area: drop train users within this many meters of a test user (default 0)")
    parser.add_argument("--seed", type=int, default=1, help="Seed for all random choices (default 1)")
    args = parser.parse_args()
    if not 0.0 < args.test_ratio < 1.0:
        parser.error("--test-ratio must be between 0 and 1")
    parts = args.by.split("+")
    if args.test_ids is not None and parts[0] not in TX_SPLITS:
        parser.error("--test-ids needs a sector or bs split")
    if args.disjoint_users and (parts[0] not in TX_SPLITS or len(parts) > 1):
        parser.error("--disjoint-users applies to a sector or bs split alone")
    if "area" not in parts and (args.guard or args.block_size != 50.0):
        parser.error("--block-size and --guard apply to an area split")
    return args


def main():
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Output directory exists: {args.out}")
    index = read_run_index(args.run)
    with open(args.run / "metadata.yaml") as f:
        run_meta = yaml.load(f, Loader=yaml.FullLoader)
    rng = np.random.default_rng(args.seed)
    parts = args.by.split("+")
    tx_method = next((p for p in parts if p in TX_SPLITS), None)
    user_method = next((p for p in parts if p in USER_SPLITS), None)
    details = {}

    tx_side = {}
    if tx_method:
        nonempty = [t for t in index["txs"] if index["rows"][t]]
        res = split_txs(index["txs"], tx_method, index["num_sectors"], args.test_ratio, rng, args.test_ids,
                        nonempty_txs=nonempty)
        tx_side = res["side"]
        details[f"test_{'sectors' if tx_method == 'sector' else 'base_stations'}"] = res["test_ids"]

    users = index["users"]
    side = None
    if user_method == "users":
        side = split_users_random(index["positions"], args.test_ratio, rng)
    elif user_method == "area":
        res = split_users_area(index["positions"], args.test_ratio, args.block_size, args.guard, rng)
        side = res["side"]
        details.update(test_blocks=res["test_blocks"], block_origin=res["block_origin"],
                       num_blocks=res["num_blocks"], num_guarded_users=res["num_guarded_users"])
    elif args.disjoint_users:
        side = np.array([tx_side[int(s)] for s in index["serving_tx"]])
    user_side = {}
    if side is not None:
        # Users at the same position (runs before 0.3.0) must not be on different sides
        details["num_users_dropped_shared_position"] = drop_split_positions(side, index["positions"])
        user_side = dict(zip(users, side.tolist()))

    split_info = dict(
        method=args.by,
        description=TEST_LABELS[f"{args.by}_disjoint" if args.disjoint_users else args.by],
        parameters=dict(test_ratio=args.test_ratio, test_ids=args.test_ids, disjoint_users=args.disjoint_users,
                        block_size=args.block_size if user_method == "area" else None,
                        guard=args.guard if user_method == "area" else None, seed=args.seed),
        **details,
        source=dict(run_dir=str(args.run.resolve()), scene_name=run_meta.get("scene_name"),
                    run_timestamp=run_meta.get("run_timestamp"), csigen_version=run_meta.get("csigen_version")),
        split_by=dict(csigen_version=__version__, git_commit=git_commit(_root),
                      command=" ".join(sys.argv), timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    summary = write_split(args.run, args.out, tx_side, user_side, split_info)
    logger.info("Train: %s channels of %s users; test: %s channels of %s users; dropped: %s channels",
                summary["num_channels"]["train"], summary["num_users"]["train"],
                summary["num_channels"]["test"], summary["num_users"]["test"], summary["num_channels_dropped"])
    logger.info("Wrote %s and %s", args.out / "train", args.out / "test")


if __name__ == "__main__":
    main()
