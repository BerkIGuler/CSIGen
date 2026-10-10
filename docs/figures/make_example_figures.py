"""
Make the example figures of the manual (docs/main.tex, section Examples) from a run.

The figures in the manual come from config/examples/eval/la_config_test.yaml:
  python scripts/run.py --config config/examples/eval/la_config_test.yaml
  python docs/figures/make_example_figures.py --run output/la_1/<run_id>
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Output files, next to this script
OUT_DIR = Path(__file__).resolve().parent
STATS_FIGURE = OUT_DIR / "example_path_stats.png"
CFR_FIGURE = OUT_DIR / "example_cfr.png"

# The example channel: of the NLoS channels of this TX file with at least this
# many paths and at least the median path gain, the one whose strongest path
# has the smallest share of the power
EXAMPLE_TX = 0
EXAMPLE_MIN_PATHS = 5


def load_stats(run: Path):
    """Concatenate path_stats and los_binary over all TX files of a run."""
    stats, los = {}, []
    for f in sorted(run.glob("channel_tx_*.npz")):
        with np.load(f, allow_pickle=True) as z:
            meta = z["metadata"].item()
        for key, values in meta["path_stats"].items():
            stats.setdefault(key, []).append(np.asarray(values))
        los.append(np.asarray(meta["los_binary"]).astype(bool))
    return {k: np.concatenate(v) for k, v in stats.items()}, np.concatenate(los)


def cdf(ax, values, label):
    values = np.sort(values[np.isfinite(values)])
    ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=label)


def plot_stats(run: Path):
    stats, los = load_stats(run)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    num_paths = stats["num_paths"]
    axes[0].hist(num_paths, bins=np.arange(0, num_paths.max() + 2) - 0.5)
    axes[0].set(xlabel="Number of paths", ylabel="Channels", title="Paths per channel")
    for mask, label in ((los, "LoS"), (~los, "NLoS")):
        cdf(axes[1], stats["delay_spread_s"][mask] * 1e9, label)
        cdf(axes[2], np.degrees(stats["asd_rad"][mask]), f"ASD, {label}")
        cdf(axes[2], np.degrees(stats["asa_rad"][mask]), f"ASA, {label}")
    axes[1].set(xlabel="Delay spread (ns)", ylabel="CDF", title="Delay spread",
                xlim=(0, np.nanpercentile(stats["delay_spread_s"] * 1e9, 99)))
    axes[2].set(xlabel="Azimuth spread (deg)", ylabel="CDF", title="Azimuth spreads")
    for ax in axes[1:]:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(STATS_FIGURE, dpi=150)
    plt.close(fig)
    print(f"{len(num_paths)} channels ({los.mean():.0%} LoS); median {np.median(num_paths):.0f} paths; wrote {STATS_FIGURE}")


def plot_cfr(run: Path):
    with np.load(run / f"channel_tx_{EXAMPLE_TX}.npz", allow_pickle=True) as z:
        h, meta = z["h"], z["metadata"].item()
    stats = meta["path_stats"]
    gain = stats["path_gain_db"]
    candidates = np.flatnonzero((np.asarray(meta["los_binary"]) == 0) & (stats["num_paths"] >= EXAMPLE_MIN_PATHS)
                                & (gain >= np.nanmedian(gain)))
    row = int(candidates[np.argmin(stats["dominant_power_fraction"][candidates])])
    # [rx_ant, 1, tx_ant, symbols, subcarriers] -> first RX antenna
    hc = h[row, 0, 0]
    mag_db = 20 * np.log10(np.abs(hc) + 1e-30)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    # Magnitude over TX antennas and subcarriers in the first symbol, and over
    # symbols and subcarriers for the first TX antenna, where the Doppler shifts show
    im = axes[0].imshow(mag_db[:, 0, :], aspect="auto", origin="lower")
    axes[0].set(xlabel="Subcarrier", ylabel="TX antenna", title="|h| (dB), first OFDM symbol")
    fig.colorbar(im, ax=axes[0])
    im = axes[1].imshow(mag_db[0].T, aspect="auto", origin="lower")
    axes[1].set(xlabel="OFDM symbol", ylabel="Subcarrier", title="|h| (dB), first TX antenna")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(CFR_FIGURE, dpi=150)
    plt.close(fig)
    speed = np.linalg.norm(np.asarray(meta["rx_velocities"])[row])
    print(f"CFR: TX {EXAMPLE_TX}, row {row}, {int(stats['num_paths'][row])} paths, "
          f"gain {gain[row]:.1f} dB, strongest path {stats['dominant_power_fraction'][row]:.0%} of the power, "
          f"delay spread {stats['delay_spread_s'][row] * 1e9:.0f} ns, speed {speed:.1f} m/s; wrote {CFR_FIGURE}")


def main():
    parser = argparse.ArgumentParser(description="Make the example figures of the manual from a run")
    parser.add_argument("--run", type=Path, required=True, help="Run directory (channel_tx_*.npz)")
    args = parser.parse_args()
    plot_stats(args.run)
    plot_cfr(args.run)


if __name__ == "__main__":
    main()
