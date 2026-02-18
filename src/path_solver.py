"""
Path solving utilities for efficient per-TX path computation.
"""

from sionna.rt import PathSolver, Paths
from typing import List, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _solve_paths_for_single_tx(
    scene,
    tx_idx: int,
    num_txs: int,
    num_sectors: int,
    num_users_per_tx: int,
    per_tx_users_only: bool = True,
    max_depth: int = 5,
    max_num_paths_per_src: int = 10**6,
    samples_per_src: int = 10**6,
    synthetic_array: bool = True,
    los: bool = True,
    specular_reflection: bool = True,
    diffuse_reflection: bool = True,
    refraction: bool = True,
    diffraction: bool = True,
    edge_diffraction: bool = True,
    diffraction_lit_region: bool = False,
    seed: int = 1,
):
    """
    Internal helper to solve paths for a single TX index.

    This contains the core logic previously inside the loop of solve_paths_per_tx.
    It is used by the streaming API to avoid keeping all Paths objects in memory.
    """
    # PathSolver is computationally expensive, so we solve for each TX separately
    ps = PathSolver()

    # Get all receivers and TXs from scene
    all_rx_names = [obj for obj in scene.receivers]
    all_tx_names = [obj for obj in scene.transmitters]

    # Map tx_idx to TX name
    bs_id = tx_idx // num_sectors
    sector_id = (tx_idx % num_sectors) + 1
    tx_name = f"BS_{bs_id}_sector_{sector_id}"

    # Determine which receivers to use for this TX
    if per_tx_users_only:
        # Only use receivers that are associated with this TX
        start_idx = tx_idx * num_users_per_tx
        end_idx = (tx_idx + 1) * num_users_per_tx
        selected_rx_names = all_rx_names[start_idx:end_idx]
        logger.info(
            "TX %s (%s): Solving paths for %s associated users (UE_%s to UE_%s)",
            tx_idx,
            tx_name,
            len(selected_rx_names),
            start_idx,
            end_idx - 1,
        )
    else:
        # Use all receivers
        selected_rx_names = all_rx_names
        logger.info(
            "TX %s (%s): Solving paths for all %s users",
            tx_idx,
            tx_name,
            len(selected_rx_names),
        )

    # Temporarily remove other receivers and TXs from scene
    rx_names_to_remove = [name for name in all_rx_names if name not in selected_rx_names]
    tx_names_to_remove = [name for name in all_tx_names if name != tx_name]

    # Store removed objects so we can restore them later
    removed_rxs = []
    removed_txs = []

    # Remove receivers
    for rx_name_to_remove in rx_names_to_remove:
        removed_rxs.append(scene.get(rx_name_to_remove))
        scene.remove(rx_name_to_remove)

    # Remove TXs
    for tx_name_to_remove in tx_names_to_remove:
        removed_txs.append(scene.get(tx_name_to_remove))
        scene.remove(tx_name_to_remove)

    try:
        logger.info(
            "Solving paths for TX %s (%s) with %s receivers",
            tx_idx,
            tx_name,
            len(selected_rx_names),
        )
        logger.info("Scene receivers: %s", len(scene.receivers))
        logger.info("Scene transmitters: %s", len(scene.transmitters))

        # Solve paths for this TX with selected receivers
        paths_tx = ps(
            scene,
            max_depth=max_depth,
            max_num_paths_per_src=max_num_paths_per_src,
            samples_per_src=samples_per_src,
            synthetic_array=synthetic_array,
            los=los,
            specular_reflection=specular_reflection,
            diffuse_reflection=diffuse_reflection,
            refraction=refraction,
            diffraction=diffraction,
            edge_diffraction=edge_diffraction,
            diffraction_lit_region=diffraction_lit_region,
            seed=seed,
        )
        return paths_tx

    finally:
        # Restore removed receivers and TXs
        for rx_obj in removed_rxs:
            scene.add(rx_obj)
        for tx_obj in removed_txs:
            scene.add(tx_obj)


def solve_paths_per_tx(
    scene,
    num_txs: int,
    num_sectors: int,
    num_users_per_tx: int,
    per_tx_users_only: bool = True,
    max_depth: int = 5,
    max_num_paths_per_src: int = 10**6,
    samples_per_src: int = 10**6,
    synthetic_array: bool = True,
    los: bool = True,
    specular_reflection: bool = True,
    diffuse_reflection: bool = True,
    refraction: bool = True,
    diffraction: bool = True,
    edge_diffraction: bool = True,
    diffraction_lit_region: bool = False,
    seed: int = 1
) -> List:
    """
    Solve paths for each TX separately (computationally efficient).
    
    This function temporarily removes other TXs and if there are any non-associated receivers, they are also removed
    from the scene for to reduce computational load on ray tracing.
    
    Parameters
    ----------
    scene : sionna.rt.Scene
        The Sionna scene object with transmitters and receivers already added
    num_txs : int
        Total number of transmitters (base stations x sectors per base station)
    num_sectors : int
        Number of sectors per base station (e.g. 1, 3, 6, etc.)
    num_users_per_tx : int
        Number of users sampled per TX
    per_tx_users_only : bool, default=True
        If True, solve paths only for users associated with each TX (sampled from the radio map).
        If False, solve for all users (but still per TX, i.e. all users are associated with each TX)
    max_depth : int, default=5
        Maximum number of ray scene interactions
    max_num_paths_per_src : int, default=10**6
        Maximum number of paths per source
    samples_per_src : int, default=10**6
        Number of samples per source
    synthetic_array : bool, default=True
        Use synthetic array for path computation (faster computation time)
    los : bool, default=True
        Include line-of-sight paths
    specular_reflection : bool, default=True
        Include specular reflections
    diffuse_reflection : bool, default=True
        Include diffuse reflections
    refraction : bool, default=True
        Include refraction
    diffraction : bool, default=True
        Include diffraction
    edge_diffraction : bool, default=True
        Include edge diffraction
    diffraction_lit_region : bool, default=False
        Include diffraction in lit region
    seed : int, default=1
        Seed for reproducibility
    
    Returns
    -------
    list
        List of Paths objects (one per TX)
    """
    paths_per_tx = []
    for tx_idx in range(num_txs):
        paths_tx = _solve_paths_for_single_tx(
            scene=scene,
            tx_idx=tx_idx,
            num_txs=num_txs,
            num_sectors=num_sectors,
            num_users_per_tx=num_users_per_tx,
            per_tx_users_only=per_tx_users_only,
            max_depth=max_depth,
            max_num_paths_per_src=max_num_paths_per_src,
            samples_per_src=samples_per_src,
            synthetic_array=synthetic_array,
            los=los,
            specular_reflection=specular_reflection,
            diffuse_reflection=diffuse_reflection,
            refraction=refraction,
            diffraction=diffraction,
            edge_diffraction=edge_diffraction,
            diffraction_lit_region=diffraction_lit_region,
            seed=seed,
        )
        paths_per_tx.append(paths_tx)

    logger.info("\\nCompleted path solving for %s TXs", len(paths_per_tx))
    return paths_per_tx


def get_doppler_stats(paths_per_tx: List[Paths]) -> Tuple[List[float], List[float], List[Tuple[float, float]]]:
    """
    Get Doppler statistics per TX: mean shift, spread (std), and (min, max) across all paths to all receivers.
    """
    mean_doppler_shifts = []
    doppler_spreads = []
    min_max_doppler_shifts = []
    for paths in paths_per_tx:
        doppler_shifts = paths.doppler.numpy()
        mean_doppler_shifts.append(np.mean(doppler_shifts))
        doppler_spreads.append(np.std(doppler_shifts))
        min_max_doppler_shifts.append((np.min(doppler_shifts), np.max(doppler_shifts)))
    return mean_doppler_shifts, doppler_spreads, min_max_doppler_shifts

def get_delay_stats(paths_per_tx: List[Paths]) -> Tuple[List[float], List[float]]:
    """
    Compute per-TX delay statistics from `paths_per_tx`, discarding invalid entries.

    - Delays are taken from `paths.tau` (seconds, Sionna convention).
    - Invalid delays are encoded as -1 in Sionna and are excluded from the statistics.
    - Some paths may have no valid delays.
    """
    mean_delays: List[float] = []
    rms_delay_spreads: List[float] = []

    for paths in paths_per_tx:
        raw = paths.tau.numpy()
        delays = np.asarray(raw, dtype=np.float64).flatten()

        # Keep only valid delays (>= 0); invalid ones are encoded as -1
        valid_mask = delays >= 0.0
        if not np.any(valid_mask):
            # No valid paths for this TX
            mean_delays.append(np.nan)
            rms_delay_spreads.append(np.nan)
            continue

        d = delays[valid_mask]
        mean_delay = float(np.mean(d))
        rms_delay_spread = float(np.sqrt(np.mean((d - mean_delay) ** 2)))

        # min and max delays removed as per instructions
        mean_delays.append(mean_delay)
        rms_delay_spreads.append(rms_delay_spread)

    return mean_delays, rms_delay_spreads


def get_num_paths_histogram(paths_per_tx: List[Paths]) -> List[List[int]]:
    """
    For each TX, compute a histogram over the number of valid paths per user,
    based on `paths.tau`.

    Returns a list of lists, one per TX index. For TX j:
      - path_count[j] is a list of length (max valid paths for that TX + 1).
      - path_count[j][i] is the (non-normalized) number of users for TX j
        that have exactly i valid paths.
    Inner list lengths can differ across TXs.
    """
    path_count: List[List[int]] = []

    for paths in paths_per_tx:
        tau = paths.tau.numpy()

        if tau.size == 0:
            hist = []
        else:
            num_users = tau.shape[0]
            valid_paths_hist = {}
            for user_idx in range(num_users):
                num_valid_paths = int(np.sum(tau[user_idx] > 0))
                if num_valid_paths not in valid_paths_hist:
                    valid_paths_hist[num_valid_paths] = 0
                valid_paths_hist[num_valid_paths] += 1
            max_n = max(valid_paths_hist.keys())
            hist = [valid_paths_hist.get(i, 0) for i in range(max_n + 1)]
        path_count.append(hist)

    return path_count