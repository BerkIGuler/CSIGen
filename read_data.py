"""
Utility script to inspect a saved channel .npz file.

Usage:
    python read_data.py --file output/<scene_name>/<timestamp>/channel_tx_0.npz

This matches the new directory layout created by sample_run.py, e.g.:
    output/boston_1/20260210_123456/channel_tx_0.npz
"""

import argparse
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read and print contents of a saved channel .npz file"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the .npz file (e.g., output/<scene_name>/<timestamp>/channel_tx_0.npz)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    npz_path = Path(args.file)

    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    print(f"Loaded file: {npz_path}")
    print("Keys:", list(data.keys()))

    # Print shapes and dtypes for common arrays
    for key in data.keys():
        val = data[key]
        if isinstance(val, np.ndarray):
            print(f"- {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"- {key}: type={type(val)}")

    # If metadata exists, print it
    if "metadata" in data:
        meta = data["metadata"].item() if hasattr(data["metadata"], "item") else data["metadata"]
        print("\nMetadata:")
        for k, v in meta.items():
            # Avoid printing very large arrays here
            if isinstance(v, np.ndarray):
                print(f"  {k}: ndarray shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
