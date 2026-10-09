"""
Path solving utilities for efficient per-TX path computation.
"""

from sionna.rt import PathSolver, Paths
from sionna.rt.constants import InteractionType
from typing import List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def set_specular_chain_table_size(size: int) -> None:
    """
    Set the minimum size of Sionna RT's specular-chain hash table per source.

    Sionna's shoot-and-bounce candidate generator stores each new (specular
    chain, receiver) pair in a hash table of size
    ``max(max_num_paths_per_src, MIN_SPEC_COUNT_SIZE)`` per source. When many
    receivers share one solve, the table fills up, new chains collide with
    stored ones and are discarded silently, so mostly NLoS paths are lost.
    ``MIN_SPEC_COUNT_SIZE`` is an internal constant of Sionna RT 2.2.0.
    """
    from sionna.rt.path_solvers.sb_candidate_generator import SBCandidateGenerator
    if not hasattr(SBCandidateGenerator, "MIN_SPEC_COUNT_SIZE"):
        raise RuntimeError("This Sionna RT version has no SBCandidateGenerator.MIN_SPEC_COUNT_SIZE.")
    SBCandidateGenerator.MIN_SPEC_COUNT_SIZE = int(size)
    logger.info("Set the specular-chain hash table size to %s per source", int(size))


def _compute_rx_valid_and_los_masks(paths: Paths) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal helper to compute both:
      - valid_mask: boolean array of shape (num_rx,), True if RX has at least
        one valid path (tau > 0)
      - rx_state_mask: integer array of shape (num_rx,) with values
            1  : RX has at least one valid pure LoS path
            0  : RX has valid paths but none purely LoS (only NLoS)
           -1  : RX has no valid paths

    This does a single pass over tau and interactions to avoid duplicate work.
    """
    tau = paths.tau.numpy()
    if tau.size == 0:
        # No paths at all: every RX is invalid (tau keeps the RX axis first)
        num_rx = tau.shape[0] if tau.ndim > 0 else 0
        return np.zeros(num_rx, dtype=bool), np.full(num_rx, -1, dtype=int)

    tau = np.asarray(tau, dtype=np.float64)
    # Valid paths per ray: same criterion as other helpers (tau > 0)
    valid_path = tau > 0

    # interactions shape: [max_depth, num_rx, ..., num_paths]
    interactions = paths.interactions.numpy()
    interactions = np.asarray(interactions, dtype=np.uint32)
    # For each path, check if all depths have InteractionType.NONE
    # Result has same shape as tau: [num_rx, ..., num_paths]
    is_los_per_path = np.all(interactions == InteractionType.NONE, axis=0)

    valid_los = valid_path & is_los_per_path

    # Collapse over all non-receiver axes to get per-RX flags
    path_axes = tuple(range(1, valid_path.ndim))
    valid_mask = np.any(valid_path, axis=path_axes)
    has_los = np.any(valid_los, axis=path_axes)

    num_rx = tau.shape[0]
    mask = np.full(num_rx, -1, dtype=int)
    mask[valid_mask] = 0
    mask[has_los] = 1
    return np.asarray(valid_mask, dtype=bool), mask


def get_valid_rx_mask(paths: Paths) -> np.ndarray:
    """
    Public helper: return boolean validity mask per RX.

    See `_compute_rx_valid_and_los_masks` for details.
    """
    valid_mask, _ = _compute_rx_valid_and_los_masks(paths)
    return valid_mask


def get_rx_los_nlos_mask(paths: Paths) -> np.ndarray:
    """
    Public helper: return integer LOS/NLOS/invalid mask per RX.

    See `_compute_rx_valid_and_los_masks` for details.
    """
    _, rx_state_mask = _compute_rx_valid_and_los_masks(paths)
    return rx_state_mask


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
    bs_id = tx_idx // num_sectors
    sector_id = (tx_idx % num_sectors) + 1
    tx_name = f"BS_{bs_id}_sector_{sector_id}"

    if per_tx_users_only:
        # Users are named UE_<tx_idx * num_users_per_tx + j> (see add_receivers_from_samples)
        start_idx = tx_idx * num_users_per_tx
        rx_names = [f"UE_{i}" for i in range(start_idx, start_idx + num_users_per_tx)]
    else:
        rx_names = list(scene.receivers)

    paths_tx, _ = solve_paths_for_receivers(
        scene,
        tx_name=tx_name,
        rx_names=rx_names,
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


def solve_paths_for_receivers(
    scene,
    tx_name: str,
    rx_names: List[str],
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
) -> Tuple[Paths, List[str]]:
    """
    Solve paths from one transmitter to the named receivers.

    All other transmitters and receivers are removed from the scene during the
    solve and added back afterwards. Receivers are selected by name, so the
    result does not depend on the order of ``scene.receivers``.

    Returns
    -------
    tuple
        (paths, row_rx_names): the Paths object and the receiver names in the
        order of its receiver axis.
    """
    ps = PathSolver()
    selected = set(rx_names)
    missing = selected - set(scene.receivers)
    if missing:
        raise ValueError(f"Receivers not in scene: {sorted(missing)[:5]}")

    removed_rxs = [scene.get(name) for name in list(scene.receivers) if name not in selected]
    removed_txs = [scene.get(name) for name in list(scene.transmitters) if name != tx_name]
    for obj in removed_rxs + removed_txs:
        scene.remove(obj.name)

    try:
        # The receiver axis of the paths follows the scene's receiver order
        row_rx_names = list(scene.receivers)
        logger.info("Solving paths for %s with %s receivers", tx_name, len(row_rx_names))
        paths = ps(
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
        return paths, row_rx_names
    finally:
        for obj in removed_rxs + removed_txs:
            scene.add(obj)


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
    Get Doppler statistics per TX: mean shift, spread (std), and (min, max).

    Only paths with tau > 0 are included. Only receivers that have at least one
    valid path are included (mask derived from paths.tau per TX).
    """
    mean_doppler_shifts = []
    doppler_spreads = []
    min_max_doppler_shifts = []
    for paths in paths_per_tx:
        tau = np.asarray(paths.tau.numpy(), dtype=np.float64)
        doppler = np.asarray(paths.doppler.numpy(), dtype=np.float64)
        # Valid receivers: at least one path with tau > 0
        valid_rx = np.any(tau > 0, axis=tuple(range(1, tau.ndim)))
        tau = tau[valid_rx, ...]
        doppler = doppler[valid_rx, ...]
        # Only include doppler for valid paths (tau > 0)
        valid_path = tau > 0
        d = doppler[valid_path]
        if d.size == 0:
            mean_doppler_shifts.append(np.nan)
            doppler_spreads.append(np.nan)
            min_max_doppler_shifts.append((np.nan, np.nan))
        else:
            mean_doppler_shifts.append(float(np.mean(d)))
            doppler_spreads.append(float(np.std(d)))
            min_max_doppler_shifts.append((float(np.min(d)), float(np.max(d))))
    return mean_doppler_shifts, doppler_spreads, min_max_doppler_shifts

def get_delay_stats(paths_per_tx: List[Paths]) -> Tuple[List[float], List[float]]:
    """
    Compute per-TX delay statistics from `paths_per_tx`, discarding invalid entries.

    - Delays are taken from `paths.tau` (seconds, Sionna convention).
    - Only paths with tau > 0 are included. Only receivers that have at least one
      valid path are included (mask derived from paths.tau per TX).
    """
    mean_delays: List[float] = []
    rms_delay_spreads: List[float] = []

    for paths in paths_per_tx:
        raw = paths.tau.numpy()
        tau = np.asarray(raw, dtype=np.float64)
        # Valid receivers: at least one path with tau > 0
        valid_rx = np.any(tau > 0, axis=tuple(range(1, tau.ndim)))
        tau = tau[valid_rx, ...]
        delays = tau.flatten()
        # Keep only valid path delays (tau > 0)
        valid_mask = delays > 0
        if not np.any(valid_mask):
            mean_delays.append(np.nan)
            rms_delay_spreads.append(np.nan)
            continue

        d = delays[valid_mask]
        mean_delay = float(np.mean(d))
        rms_delay_spread = float(np.sqrt(np.mean((d - mean_delay) ** 2)))
        mean_delays.append(mean_delay)
        rms_delay_spreads.append(rms_delay_spread)

    return mean_delays, rms_delay_spreads


def get_num_paths_histogram(
    paths_per_tx: List[Paths],
    valid_rx_mask_per_tx: Optional[List[np.ndarray]] = None,
) -> List[List[int]]:
    """
    For each TX, compute a histogram over the number of valid paths per user,
    based on `paths.tau`.

    If valid_rx_mask_per_tx is provided, only receivers with mask True are
    included (e.g. to match CFR filtered to valid channels).

    Returns a list of lists, one per TX index. For TX j:
      - path_count[j] is a list of length (max valid paths for that TX + 1).
      - path_count[j][i] is the (non-normalized) number of users for TX j
        that have exactly i valid paths.
    Inner list lengths can differ across TXs.
    """
    path_count: List[List[int]] = []

    for tx_idx, paths in enumerate(paths_per_tx):
        tau = paths.tau.numpy()

        if tau.size == 0:
            hist = []
        else:
            num_users = tau.shape[0]
            if valid_rx_mask_per_tx is not None and tx_idx < len(valid_rx_mask_per_tx):
                mask = np.asarray(valid_rx_mask_per_tx[tx_idx], dtype=bool)
                user_indices = np.where(mask)[0]
            else:
                user_indices = np.arange(num_users)
            valid_paths_hist = {}
            for user_idx in user_indices:
                num_valid_paths = int(np.sum(tau[user_idx] > 0))
                if num_valid_paths not in valid_paths_hist:
                    valid_paths_hist[num_valid_paths] = 0
                valid_paths_hist[num_valid_paths] += 1
            max_n = max(valid_paths_hist.keys()) if valid_paths_hist else 0
            hist = [valid_paths_hist.get(i, 0) for i in range(max_n + 1)]
        path_count.append(hist)

    return path_count