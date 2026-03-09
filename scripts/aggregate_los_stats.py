"""
Aggregate LoS statistics across all channel files for a given city.

This script mirrors the core logic in `examples/visualize_dataset_by_los.ipynb`,
but instead of operating on a single `channel_tx_*.npz` file it:

- Loads **all** `.npz` channel files for a given city/run
- Aggregates LoS masks and per-sample channel power
- Computes global LoS percentage and power statistics
- Saves aggregated histograms (linear and dB) to disk
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate LoS statistics over all channel files for a city."
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        help="City name (e.g., boston_1, chicago_1).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/opt/shared/datasets/CSIGen/TemporalWiMAE"),
        help="Root directory containing per-city runs "
        "(default: /opt/shared/datasets/CSIGen/TemporalWiMAE).",
    )
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default=None,
        help=(
            "Timestamp subdirectory under the city (e.g., 20260302_121714). "
            "If omitted, the lexicographically latest subdirectory is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("aggregate_plots"),
        help="Directory where plots will be saved (created if needed).",
    )
    return parser.parse_args()


def find_data_dir(dataset_root: Path, city: str, run_timestamp: str | None) -> Path:
    city_dir = dataset_root / city
    if not city_dir.is_dir():
        raise FileNotFoundError(f"City directory not found: {city_dir}")

    if run_timestamp is not None:
        data_dir = city_dir / run_timestamp
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {data_dir}")
        return data_dir

    # Auto-select latest timestamp directory
    candidates = sorted(
        [p for p in city_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {city_dir}")

    return candidates[-1]


def compute_per_sample_power(h: np.ndarray) -> np.ndarray:
    """
    Compute per-sample channel power, matching the notebook's logic.

    The first axis is assumed to be the sample axis; all remaining axes
    are averaged over.
    """
    h = np.asarray(h)
    if h.ndim < 2:
        raise ValueError(f"Expected h.ndim >= 2, got shape {h.shape}")

    sample_axes = tuple(range(1, h.ndim))
    power = np.mean(np.abs(h) ** 2, axis=sample_axes)
    return power


def aggregate_los_stats(npz_files: List[Path]) -> dict:
    all_los_powers: List[np.ndarray] = []
    total_los = 0
    total_samples = 0

    for npz_file in npz_files:
        with np.load(npz_file, allow_pickle=True) as data:
            h = np.squeeze(data["h"])
            meta = data["metadata"].item()

            los_binary = np.asarray(meta["los_binary"]).astype(bool)

            per_sample_power = compute_per_sample_power(h)
            if per_sample_power.shape[0] != los_binary.shape[0]:
                raise ValueError(
                    f"Sample dimension mismatch for {npz_file.name}: "
                    f"power.shape[0]={per_sample_power.shape[0]}, "
                    f"los_binary.shape[0]={los_binary.shape[0]}"
                )

            all_los_powers.append(per_sample_power[los_binary])

            total_los += int(los_binary.sum())
            total_samples += int(los_binary.size)

    if not all_los_powers:
        raise RuntimeError("No LoS samples found in any file.")

    los_powers = np.concatenate(all_los_powers, axis=0)

    los_percentage = 100.0 * total_los / max(total_samples, 1)
    los_mean_linear = float(los_powers.mean())
    los_mean_db = 10.0 * np.log10(los_mean_linear)

    return {
        "los_powers": los_powers,
        "total_los": total_los,
        "total_samples": total_samples,
        "los_percentage": los_percentage,
        "los_mean_linear": los_mean_linear,
        "los_mean_db": los_mean_db,
    }


def plot_histograms(
    los_powers: np.ndarray,
    los_percentage: float,
    los_mean_linear: float,
    los_mean_db: float,
    city: str,
    run_timestamp: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # dB histogram
    los_powers_db = 10.0 * np.log10(los_powers)
    plt.figure(figsize=(8, 5))
    plt.hist(los_powers_db, bins=100, density=True)
    plt.xlabel("Channel Power (dB)")
    plt.ylabel("Density")
    plt.title(
        f"LoS Channel Power Distribution (All TXs)\n"
        f"{city}, {run_timestamp}  |  LoS: {los_percentage:.1f}%"
    )
    plt.axvline(
        los_mean_db,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean power: {los_mean_db:.2f} dB",
    )
    plt.legend()
    plt.tight_layout()
    db_path = output_dir / f"{city}_{run_timestamp}_los_power_db_all.png"
    plt.savefig(db_path, dpi=200)
    plt.close()

    # Linear histogram
    plt.figure(figsize=(8, 5))
    plt.hist(los_powers, bins=100, density=True)
    plt.xlabel("Channel Power (linear)")
    plt.ylabel("Density")
    plt.title(
        f"LoS Channel Power Distribution (linear, All TXs)\n"
        f"{city}, {run_timestamp}  |  LoS: {los_percentage:.1f}%"
    )
    plt.axvline(
        los_mean_linear,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean power: {los_mean_linear:.3e}",
    )
    plt.legend()
    plt.tight_layout()
    lin_path = output_dir / f"{city}_{run_timestamp}_los_power_linear_all.png"
    plt.savefig(lin_path, dpi=200)
    plt.close()

    print(f"Saved dB histogram to:     {db_path}")
    print(f"Saved linear histogram to: {lin_path}")


def main() -> None:
    args = parse_args()

    data_dir = find_data_dir(args.dataset_root, args.city, args.run_timestamp)
    run_timestamp = data_dir.name

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    print(f"Found {len(npz_files)} .npz files in {data_dir}")

    stats = aggregate_los_stats(npz_files)

    print("")
    print(f"City:           {args.city}")
    print(f"Run timestamp:  {run_timestamp}")
    print(f"Total samples:  {stats['total_samples']}")
    print(f"Total LoS:      {stats['total_los']} "
          f"({stats['los_percentage']:.2f}%)")
    print(f"Mean LoS power: {stats['los_mean_linear']:.3e} (linear), "
          f"{stats['los_mean_db']:.2f} dB")
    print("")

    plot_histograms(
        los_powers=stats["los_powers"],
        los_percentage=stats["los_percentage"],
        los_mean_linear=stats["los_mean_linear"],
        los_mean_db=stats["los_mean_db"],
        city=args.city,
        run_timestamp=run_timestamp,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

