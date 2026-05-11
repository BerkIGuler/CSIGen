"""
Aggregate LoS statistics across all channel files for a given city.

This script mirrors the core logic in `examples/visualize_dataset_by_los.ipynb`,
but instead of operating on a single `channel_tx_*.npz` file it:

- Loads **all** `.npz` channel files for a given city/run
- Aggregates LoS masks, per-sample channel power, and TX–RX distances
- Computes global LoS percentage and power statistics
- Saves aggregated histograms (linear and dB) and CDFs (distance, power linear, power dB) to disk
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Cities used for "all scenarios" combined CDF figures (8 curves: LoS + NLoS per city)
ALL_CITIES = ["boston_1", "nyc_1", "sf_1", "chicago_1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate LoS statistics over all channel files for a city (or all cities)."
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="Single city name (e.g., boston_1). Ignored if --all-cities is set.",
    )
    parser.add_argument(
        "--all-cities",
        action="store_true",
        help=(
            "Run for all scenarios (boston_1, nyc_1, sf_1, chicago_1), save per-city figures "
            "and additionally one combined CDF figure per type (distance, power linear, power dB) "
            "with 8 curves (LoS + NLoS for each city)."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/berkay/Desktop/research/datasets/CSIGen/TemporalWiMAE/ood_test_35"),
        help=(
            "Root directory containing per-city channel files, matching "
            "`examples/visualize_dataset_by_los.ipynb` "
            "(default: /opt/shared/datasets/CSIGen/TemporalWiMAE)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("aggregate_plots"),
        help="Directory where plots will be saved (created if needed).",
    )
    args = parser.parse_args()
    if not args.all_cities and args.city is None:
        parser.error("Either --city or --all-cities is required.")
    return args


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


def compute_per_sample_distances(tx_position: np.ndarray, rx_positions: np.ndarray) -> np.ndarray:
    """Compute TX–RX Euclidean distance per sample (meters)."""
    tx = np.asarray(tx_position).reshape(-1)
    rx = np.asarray(rx_positions)
    if rx.ndim == 1:
        rx = rx.reshape(1, -1)
    return np.linalg.norm(rx - tx, axis=1)


def aggregate_los_stats(npz_files: List[Path]) -> dict:
    all_los_powers: List[np.ndarray] = []
    all_nlos_powers: List[np.ndarray] = []
    all_los_distances: List[np.ndarray] = []
    all_nlos_distances: List[np.ndarray] = []
    total_los = 0
    total_nlos = 0
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
            all_nlos_powers.append(per_sample_power[~los_binary])

            tx_position = meta.get("tx_position")
            rx_positions = meta.get("rx_positions")
            if tx_position is not None and rx_positions is not None:
                distances = compute_per_sample_distances(tx_position, rx_positions)
                all_los_distances.append(distances[los_binary])
                all_nlos_distances.append(distances[~los_binary])

            num_los = int(los_binary.sum())
            num_samples = int(los_binary.size)
            total_los += num_los
            total_nlos += num_samples - num_los
            total_samples += num_samples

    if total_samples == 0:
        raise RuntimeError("No samples found in any file.")

    los_powers = np.concatenate(all_los_powers, axis=0) if total_los > 0 else np.array([])
    nlos_powers = np.concatenate(all_nlos_powers, axis=0) if total_nlos > 0 else np.array([])
    los_distances = np.concatenate(all_los_distances, axis=0) if all_los_distances else np.array([])
    nlos_distances = np.concatenate(all_nlos_distances, axis=0) if all_nlos_distances else np.array([])

    los_percentage = 100.0 * total_los / total_samples
    nlos_percentage = 100.0 * total_nlos / total_samples

    los_mean_linear = float(los_powers.mean()) if total_los > 0 else float("nan")
    nlos_mean_linear = float(nlos_powers.mean()) if total_nlos > 0 else float("nan")

    los_mean_db = 10.0 * np.log10(los_mean_linear) if total_los > 0 else float("nan")
    nlos_mean_db = 10.0 * np.log10(nlos_mean_linear) if total_nlos > 0 else float("nan")

    return {
        "los_powers": los_powers,
        "nlos_powers": nlos_powers,
        "los_distances": los_distances,
        "nlos_distances": nlos_distances,
        "total_los": total_los,
        "total_nlos": total_nlos,
        "total_samples": total_samples,
        "los_percentage": los_percentage,
        "nlos_percentage": nlos_percentage,
        "los_mean_linear": los_mean_linear,
        "nlos_mean_linear": nlos_mean_linear,
        "los_mean_db": los_mean_db,
        "nlos_mean_db": nlos_mean_db,
    }


def plot_histograms(
    powers: np.ndarray,
    percentage: float,
    mean_linear: float,
    mean_db: float,
    city: str,
    label: str,
    suffix: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # dB histogram
    powers_db = 10.0 * np.log10(powers)
    plt.figure(figsize=(8, 5))
    plt.hist(powers_db, bins=100, density=True)
    plt.xlabel(f"{label} Channel Power (dB)")
    plt.ylabel("Density")
    plt.title(
        f"{label} Channel Power Distribution (All TXs)\n"
        f"{city}  |  {label}: {percentage:.1f}%"
    )
    plt.axvline(
        mean_db,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean power: {mean_db:.2f} dB",
    )
    plt.legend()
    plt.tight_layout()
    db_path = output_dir / f"{city}_{suffix}_power_db_all.png"
    plt.savefig(db_path, dpi=200)
    plt.close()

    # Linear histogram
    plt.figure(figsize=(8, 5))
    plt.hist(powers, bins=100, density=True)
    plt.xlabel(f"{label} Channel Power (linear)")
    plt.ylabel("Density")
    plt.title(
        f"{label} Channel Power Distribution (linear, All TXs)\n"
        f"{city}  |  {label}: {percentage:.1f}%"
    )
    plt.axvline(
        mean_linear,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean power: {mean_linear:.3e}",
    )
    plt.legend()
    plt.tight_layout()
    lin_path = output_dir / f"{city}_{suffix}_power_linear_all.png"
    plt.savefig(lin_path, dpi=200)
    plt.close()

    print(f"Saved dB histogram to:     {db_path}")
    print(f"Saved linear histogram to: {lin_path}")


def plot_distance_cdf(
    los_distances: np.ndarray,
    nlos_distances: np.ndarray,
    los_percentage: float,
    nlos_percentage: float,
    city: str,
    output_dir: Path,
) -> None:
    """Save distance CDF for the city (LoS and NLoS on same plot)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    has_los = los_distances.size > 0
    has_nlos = nlos_distances.size > 0
    if not has_los and not has_nlos:
        return

    plt.figure(figsize=(8, 5))
    if has_los:
        sorted_los = np.sort(los_distances[np.isfinite(los_distances)])
        n = sorted_los.size
        plt.plot(sorted_los, np.arange(1, n + 1, dtype=float) / n, label=f"LoS ({los_percentage:.1f}%)")
    if has_nlos:
        sorted_nlos = np.sort(nlos_distances[np.isfinite(nlos_distances)])
        n = sorted_nlos.size
        plt.plot(sorted_nlos, np.arange(1, n + 1, dtype=float) / n, label=f"NLoS ({nlos_percentage:.1f}%)")
    plt.xlabel("TX–RX distance (m)")
    plt.ylabel("CDF")
    plt.title(f"Distance CDF (All TXs)\n{city}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{city}_distance_cdf.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved distance CDF to: {path}")


def plot_power_cdfs_combined(
    los_powers: np.ndarray,
    nlos_powers: np.ndarray,
    los_percentage: float,
    nlos_percentage: float,
    city: str,
    output_dir: Path,
) -> None:
    """Save power CDF in linear and dB with LoS and NLoS on the same figure (like distance CDF)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    los_powers = np.asarray(los_powers)
    nlos_powers = np.asarray(nlos_powers)
    los_powers = los_powers[np.isfinite(los_powers) & (los_powers > 0)]
    nlos_powers = nlos_powers[np.isfinite(nlos_powers) & (nlos_powers > 0)]
    has_los = los_powers.size > 0
    has_nlos = nlos_powers.size > 0
    if not has_los and not has_nlos:
        return

    # Linear power CDF (LoS + NLoS on same plot)
    plt.figure(figsize=(8, 5))
    if has_los:
        sorted_los = np.sort(los_powers)
        n = sorted_los.size
        plt.plot(sorted_los, np.arange(1, n + 1, dtype=float) / n, label=f"LoS ({los_percentage:.1f}%)")
    if has_nlos:
        sorted_nlos = np.sort(nlos_powers)
        n = sorted_nlos.size
        plt.plot(sorted_nlos, np.arange(1, n + 1, dtype=float) / n, label=f"NLoS ({nlos_percentage:.1f}%)")
    plt.xlabel("Channel Power (linear)")
    plt.ylabel("CDF")
    plt.title(f"Channel Power CDF (linear, All TXs)\n{city}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    lin_path = output_dir / f"{city}_power_cdf_linear.png"
    plt.savefig(lin_path, dpi=200)
    plt.close()
    print(f"Saved power CDF (linear) to: {lin_path}")

    # dB power CDF (LoS + NLoS on same plot)
    plt.figure(figsize=(8, 5))
    if has_los:
        los_db = 10.0 * np.log10(los_powers)
        sorted_los = np.sort(los_db)
        n = sorted_los.size
        plt.plot(sorted_los, np.arange(1, n + 1, dtype=float) / n, label=f"LoS ({los_percentage:.1f}%)")
    if has_nlos:
        nlos_db = 10.0 * np.log10(nlos_powers)
        sorted_nlos = np.sort(nlos_db)
        n = sorted_nlos.size
        plt.plot(sorted_nlos, np.arange(1, n + 1, dtype=float) / n, label=f"NLoS ({nlos_percentage:.1f}%)")
    plt.xlabel("Channel Power (dB)")
    plt.ylabel("CDF")
    plt.title(f"Channel Power CDF (dB, All TXs)\n{city}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    db_path = output_dir / f"{city}_power_cdf_db.png"
    plt.savefig(db_path, dpi=200)
    plt.close()
    print(f"Saved power CDF (dB) to: {db_path}")


def find_data_dir(dataset_root: Path, city: str) -> Path:
    """
    Locate the directory containing channel files for a given city.

    This matches the layout used in `examples/visualize_dataset_by_los.ipynb`:
        data_dir = /opt/shared/datasets/CSIGen/<dataset_name>/<city>
    """
    data_dir = dataset_root / city
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return data_dir


def _cdf_curve_sorted(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sorted_values, cdf_y) for finite values; empty arrays if no data."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([])
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    cdf_y = np.arange(1, n + 1, dtype=float) / n
    return sorted_vals, cdf_y


def plot_all_cities_distance_cdf(
    all_stats: List[Tuple[str, dict]],
    output_dir: Path,
) -> None:
    """One figure: distance CDF with 8 curves (LoS + NLoS for each city)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for city, stats in all_stats:
        los_d, los_y = _cdf_curve_sorted(stats["los_distances"])
        nlos_d, nlos_y = _cdf_curve_sorted(stats["nlos_distances"])
        if los_d.size > 0:
            plt.plot(los_d, los_y, label=f"{city} LoS ({stats['los_percentage']:.1f}%)")
        if nlos_d.size > 0:
            plt.plot(nlos_d, nlos_y, label=f"{city} NLoS ({stats['nlos_percentage']:.1f}%)")
    plt.xlabel("TX–RX distance (m)")
    plt.ylabel("CDF")
    plt.title("Distance CDF (all scenarios)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "all_cities_distance_cdf.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved combined distance CDF to: {path}")


def plot_all_cities_power_cdf_linear(
    all_stats: List[Tuple[str, dict]],
    output_dir: Path,
) -> None:
    """One figure: power CDF (linear) with 8 curves (LoS + NLoS for each city)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for city, stats in all_stats:
        los_p = np.asarray(stats["los_powers"])
        nlos_p = np.asarray(stats["nlos_powers"])
        los_p = los_p[np.isfinite(los_p) & (los_p > 0)]
        nlos_p = nlos_p[np.isfinite(nlos_p) & (nlos_p > 0)]
        los_x, los_y = _cdf_curve_sorted(los_p)
        nlos_x, nlos_y = _cdf_curve_sorted(nlos_p)
        if los_x.size > 0:
            plt.plot(los_x, los_y, label=f"{city} LoS ({stats['los_percentage']:.1f}%)")
        if nlos_x.size > 0:
            plt.plot(nlos_x, nlos_y, label=f"{city} NLoS ({stats['nlos_percentage']:.1f}%)")
    plt.xlabel("Channel power (linear)")
    plt.ylabel("CDF")
    plt.title("Channel power CDF, linear (all scenarios)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "all_cities_power_cdf_linear.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved combined power CDF (linear) to: {path}")


def plot_all_cities_power_cdf_db(
    all_stats: List[Tuple[str, dict]],
    output_dir: Path,
) -> None:
    """One figure: power CDF (dB) with 8 curves (LoS + NLoS for each city)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for city, stats in all_stats:
        los_p = np.asarray(stats["los_powers"])
        nlos_p = np.asarray(stats["nlos_powers"])
        los_p = los_p[np.isfinite(los_p) & (los_p > 0)]
        nlos_p = nlos_p[np.isfinite(nlos_p) & (nlos_p > 0)]
        los_db = 10.0 * np.log10(los_p) if los_p.size > 0 else np.array([])
        nlos_db = 10.0 * np.log10(nlos_p) if nlos_p.size > 0 else np.array([])
        los_x, los_y = _cdf_curve_sorted(los_db)
        nlos_x, nlos_y = _cdf_curve_sorted(nlos_db)
        if los_x.size > 0:
            plt.plot(los_x, los_y, label=f"{city} LoS ({stats['los_percentage']:.1f}%)")
        if nlos_x.size > 0:
            plt.plot(nlos_x, nlos_y, label=f"{city} NLoS ({stats['nlos_percentage']:.1f}%)")
    plt.xlabel("Channel power (dB)")
    plt.ylabel("CDF")
    plt.title("Channel power CDF, dB (all scenarios)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "all_cities_power_cdf_db.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved combined power CDF (dB) to: {path}")


def _run_one_city(
    city: str,
    dataset_root: Path,
    output_dir: Path,
) -> dict:
    """Aggregate stats for one city and save per-city figures. Return stats dict."""
    data_dir = find_data_dir(dataset_root, city)
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    print(f"Found {len(npz_files)} .npz files in {data_dir}")
    stats = aggregate_los_stats(npz_files)

    print("")
    print(f"City:            {city}")
    print(f"Total samples:   {stats['total_samples']}")
    print(f"Total LoS:       {stats['total_los']} ({stats['los_percentage']:.2f}%)")
    print(f"Total NLoS:      {stats['total_nlos']} ({stats['nlos_percentage']:.2f}%)")
    print(f"Mean LoS power:  {stats['los_mean_linear']:.3e} (linear), {stats['los_mean_db']:.2f} dB")
    print(f"Mean NLoS power: {stats['nlos_mean_linear']:.3e} (linear), {stats['nlos_mean_db']:.2f} dB")
    print("")

    if stats["total_los"] > 0:
        plot_histograms(
            powers=stats["los_powers"],
            percentage=stats["los_percentage"],
            mean_linear=stats["los_mean_linear"],
            mean_db=stats["los_mean_db"],
            city=city,
            label="LoS",
            suffix="los",
            output_dir=output_dir,
        )
    if stats["total_nlos"] > 0:
        plot_histograms(
            powers=stats["nlos_powers"],
            percentage=stats["nlos_percentage"],
            mean_linear=stats["nlos_mean_linear"],
            mean_db=stats["nlos_mean_db"],
            city=city,
            label="NLoS",
            suffix="nlos",
            output_dir=output_dir,
        )
    if stats["total_los"] > 0 or stats["total_nlos"] > 0:
        plot_power_cdfs_combined(
            los_powers=stats["los_powers"],
            nlos_powers=stats["nlos_powers"],
            los_percentage=stats["los_percentage"],
            nlos_percentage=stats["nlos_percentage"],
            city=city,
            output_dir=output_dir,
        )
    if stats["los_distances"].size > 0 or stats["nlos_distances"].size > 0:
        plot_distance_cdf(
            los_distances=stats["los_distances"],
            nlos_distances=stats["nlos_distances"],
            los_percentage=stats["los_percentage"],
            nlos_percentage=stats["nlos_percentage"],
            city=city,
            output_dir=output_dir,
        )
    return stats


def main() -> None:
    args = parse_args()

    if args.all_cities:
        all_stats: List[Tuple[str, dict]] = []
        for city in ALL_CITIES:
            try:
                stats = _run_one_city(city, args.dataset_root, args.output_dir)
                all_stats.append((city, stats))
            except FileNotFoundError as e:
                print(f"Skipping {city}: {e}")
        if all_stats:
            plot_all_cities_distance_cdf(all_stats, args.output_dir)
            plot_all_cities_power_cdf_linear(all_stats, args.output_dir)
            plot_all_cities_power_cdf_db(all_stats, args.output_dir)
        return

    _run_one_city(args.city, args.dataset_root, args.output_dir)


if __name__ == "__main__":
    main()

