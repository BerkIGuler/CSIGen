"""
Report the percentage of LoS channels for each subfolder of a dataset root.

Each subfolder is treated as one scenario/city and is expected to contain
`*.npz` channel files whose `metadata` field includes a `los_binary` array.

Edit DATASET_ROOT below to point at the dataset you want to summarize.
"""

from pathlib import Path

import numpy as np

DATASET_ROOT = Path("/home/berkay/Desktop/research/datasets/CSIGen/TemporalWiMAE/test_28")


def los_stats_for_folder(folder: Path) -> tuple[int, int]:
    """Return (total_los, total_samples) aggregated over all .npz files in folder."""
    total_los = 0
    total_samples = 0
    for npz_file in sorted(folder.glob("*.npz")):
        with np.load(npz_file, allow_pickle=True) as data:
            meta = data["metadata"].item()
            los_binary = np.asarray(meta["los_binary"]).astype(bool)
        total_los += int(los_binary.sum())
        total_samples += int(los_binary.size)
    return total_los, total_samples


def main() -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    subfolders = sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir())
    if not subfolders:
        print(f"No subfolders found in {DATASET_ROOT}")
        return

    name_width = max(len(p.name) for p in subfolders)
    header = f"{'subfolder'.ljust(name_width)}  {'LoS %':>7}  {'LoS':>10}  {'Total':>10}"
    print(f"Dataset root: {DATASET_ROOT}")
    print(header)
    print("-" * len(header))

    grand_los = 0
    grand_total = 0
    for sub in subfolders:
        total_los, total_samples = los_stats_for_folder(sub)
        if total_samples == 0:
            print(f"{sub.name.ljust(name_width)}  {'n/a':>7}  {0:>10}  {0:>10}  (no .npz files)")
            continue
        pct = 100.0 * total_los / total_samples
        print(f"{sub.name.ljust(name_width)}  {pct:>6.2f}%  {total_los:>10}  {total_samples:>10}")
        grand_los += total_los
        grand_total += total_samples

    if grand_total > 0:
        overall_pct = 100.0 * grand_los / grand_total
        print("-" * len(header))
        print(
            f"{'OVERALL'.ljust(name_width)}  {overall_pct:>6.2f}%  "
            f"{grand_los:>10}  {grand_total:>10}"
        )


if __name__ == "__main__":
    main()
