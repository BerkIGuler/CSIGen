"""
Commands of CSIGen.

  csigen-run --config <config.yaml>
      Generates the channels of a config and writes the run to output/<scene_name>/<timestamp>/
      (channel_tx_*.npz and metadata.yaml), relative to the working directory.

  csigen-split --run <run> --out <out> --by <method> [options]
      Splits a generated run into a train run and a test run.

Split methods (combine a user split and a TX split with --by, e.g. "sector+area"):
  users   random users; all channels of a user go to the same side
  area    users by area: square blocks of the scene, with an optional guard
  sector  N of the M sectors (TXs) are test
  bs      N of the base stations (all their sectors) are test

The split output directory gets train/ and test/, each a run in the usual format
(channel_tx_*.npz + metadata.yaml) plus split.yaml, which records the method,
its parameters, the chosen sectors, base stations or blocks, and the counts.

Example:
  csigen-split --run output/chicago_1/20261009_100551 \\
      --out output/chicago_1/20261009_100551_split_bs --by bs --test-ratio 0.2 --seed 1

csigen-run imports the modules that need Sionna RT after it parses its arguments, so that
--help works without Sionna RT.
"""

import argparse
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from csigen import __version__
from csigen.split import (
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

logger = logging.getLogger(__name__)

# Root of the CSIGen clone, for the git commit in split.yaml (with an editable install)
_REPO = Path(__file__).resolve().parent.parent
_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SPLIT_METHODS = ("users", "area", "sector", "bs", "sector+users", "sector+area", "bs+users", "bs+area")


def parse_run_args(argv=None):
    """Parse the arguments of csigen-run."""
    parser = argparse.ArgumentParser(
        prog="csigen-run",
        description="Generate channels from a YAML configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    return parser.parse_args(argv)


def run_main(argv=None):
    """Generate the channels of a config and save the run (csigen-run)."""
    args = parse_run_args(argv)
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

    try:
        from csigen.channel_generator import generate_channels
        from csigen.channel import save_channel_data
    except ModuleNotFoundError as e:
        raise SystemExit(f"csigen-run needs Sionna RT ({e}). Install the extra rt: pip install -e \".[rt]\"") from e
    from csigen.config_validator import load_validated_config

    config_path = Path(args.config)

    logger.info(f"Loading and validating configuration from {config_path}")
    validated_config = load_validated_config(config_path)

    # Derive scene name from the directory under 'scenes'
    scene_xml_path = Path(validated_config['scene_xml_path'])
    scene_name = scene_xml_path.parent.name or scene_xml_path.stem

    # Create a timestamped run directory: output/<scene_name>/<YYYYmmdd_HHMMSS>
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / scene_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting channel generation (streaming per TX)...")

    # Global summary information to be filled as we stream results
    num_txs = None
    users_per_tx = None
    total_users = None
    num_sectors = None
    array_info = {}
    cfr_shapes = []
    cfr_dtypes = []

    for item in generate_channels(validated_config):
        tx_idx = item['tx_idx']
        h_tx = item['h_tx']
        tx_metadata = item['tx_metadata']

        # Initialize global summary info from first TX
        if num_txs is None:
            num_txs = int(tx_metadata['num_txs'])
            users_per_tx = [int(n) for n in tx_metadata['users_per_tx']]
            total_users = int(tx_metadata['total_users'])
            num_sectors = int(tx_metadata['num_sectors'])
            array_info = {key: tx_metadata[key] for key in ('cfr_axes', 'tx_array', 'rx_array')}

        cfr_shapes.append(list(tx_metadata['cfr_shape']))
        cfr_dtypes.append(tx_metadata['cfr_dtype'])

        output_path = output_dir / f"channel_tx_{tx_idx}.npz"
        # Save channel data + metadata (including positions)
        save_channel_data(
            h=h_tx,
            output_path=output_path,
            metadata=tx_metadata,
            path_data=item['path_data'],
        )
        num_valid = tx_metadata['num_valid_channels']
        logger.info(
            "Saved channel data for TX %s (%s) to %s (%s valid channels)",
            tx_idx, tx_metadata['tx_name'], output_path, num_valid,
        )

    # Also save combined metadata summary (including the config and run info)
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml_metadata = {
            'csigen_version': __version__,
            'scene_name': scene_name,
            'run_timestamp': timestamp,
            'num_txs': int(num_txs) if num_txs is not None else 0,
            'users_per_tx': users_per_tx or [],
            'total_users': int(total_users) if total_users is not None else 0,
            'num_sectors': int(num_sectors) if num_sectors is not None else 0,
            'cfr_per_tx_shapes': cfr_shapes,
            'cfr_per_tx_dtypes': cfr_dtypes,
            **array_info,
            'config': validated_config,
        }
        yaml.dump(yaml_metadata, f, default_flow_style=False)

    logger.info(f"Saved metadata summary to {metadata_path}")
    logger.info("Channel generation and saving complete!")
    logger.info(f"Output directory: {output_dir}")


def parse_split_args(argv=None):
    """Parse the arguments of csigen-split."""
    parser = argparse.ArgumentParser(prog="csigen-split", description="Split a generated run into train and test runs")
    parser.add_argument("--run", type=Path, required=True, help="Run directory (channel_tx_*.npz + metadata.yaml)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory; train/ and test/ are created in it")
    parser.add_argument("--by", choices=SPLIT_METHODS, required=True, help="Split method")
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
    args = parser.parse_args(argv)
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


def split_main(argv=None):
    """Split a generated run into a train run and a test run (csigen-split)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_split_args(argv)
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
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
        split_by=dict(csigen_version=__version__,
                      git_commit=git_commit(_REPO) if (_REPO / ".git").exists() else None,
                      command=shlex.join(["csigen-split", *argv]), timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    summary = write_split(args.run, args.out, tx_side, user_side, split_info)
    logger.info("Train: %s channels of %s users; test: %s channels of %s users; dropped: %s channels",
                summary["num_channels"]["train"], summary["num_users"]["train"],
                summary["num_channels"]["test"], summary["num_users"]["test"], summary["num_channels_dropped"])
    logger.info("Wrote %s and %s", args.out / "train", args.out / "test")


if __name__ == "__main__":
    # The module has two commands, so it has no default action
    sys.exit("Use the commands csigen-run and csigen-split, which the install adds")
