"""
Path solving utilities for efficient per-TX path computation.
"""

from sionna.rt import PathSolver, Paths
from sionna.rt.constants import InteractionType
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple
import logging

import drjit as dr
import numpy as np

from src.receivers import rx_names_for_tx

logger = logging.getLogger(__name__)

# Arguments of Sionna's PathSolver that the solvers here pass through
_SOLVER_ARGS = (
    "max_depth", "max_num_paths_per_src", "samples_per_src", "synthetic_array",
    "los", "specular_reflection", "diffuse_reflection", "refraction",
    "diffraction", "edge_diffraction", "diffraction_lit_region", "seed",
)

# Sionna's MIN_SPEC_COUNT_SIZE before the first change
_DEFAULT_SPEC_COUNT_SIZE = None


def set_specular_chain_table_size(size: Optional[int]) -> None:
    """
    Set the minimum size of Sionna RT's specular-chain hash table per source.

    Sionna's shoot-and-bounce candidate generator stores each new (specular
    chain, receiver) pair in a hash table of size
    ``max(max_num_paths_per_src, MIN_SPEC_COUNT_SIZE)`` per source. When many
    receivers share one solve, the table fills up, new chains collide with
    stored ones and are discarded silently, so mostly NLoS paths are lost.
    ``MIN_SPEC_COUNT_SIZE`` is an internal constant of Sionna RT 2.2.0 and is
    shared by all solves in the process; None restores Sionna's default.
    """
    from sionna.rt.path_solvers.sb_candidate_generator import SBCandidateGenerator
    global _DEFAULT_SPEC_COUNT_SIZE
    if not hasattr(SBCandidateGenerator, "MIN_SPEC_COUNT_SIZE"):
        raise RuntimeError("This Sionna RT version has no SBCandidateGenerator.MIN_SPEC_COUNT_SIZE.")
    if _DEFAULT_SPEC_COUNT_SIZE is None:
        _DEFAULT_SPEC_COUNT_SIZE = SBCandidateGenerator.MIN_SPEC_COUNT_SIZE
    SBCandidateGenerator.MIN_SPEC_COUNT_SIZE = int(_DEFAULT_SPEC_COUNT_SIZE if size is None else size)
    logger.info("Set the specular-chain hash table size to %s per source", SBCandidateGenerator.MIN_SPEC_COUNT_SIZE)


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



class _CandidateCounter:
    """
    Wrap Sionna's candidate generator to record how full the path buffer got.

    Sionna stores at most ``max_num_paths_per_src`` candidate paths per source
    and discards further candidates silently. ``max_fraction`` is the largest
    number of stored candidates of a source divided by that cap; at 1.0 or
    above, candidates were discarded.
    """

    def __init__(self, generator):
        self._generator = generator
        self.max_fraction = 0.0

    def __call__(self, **kwargs):
        buffer = self._generator(**kwargs)
        num_stored = int(np.array(buffer.paths_counter)[0])
        num_sources = dr.width(kwargs["src_positions"])
        if num_sources > 1 and num_stored > 0:
            src_indices = np.array(buffer._src_indices)[:num_stored]  # internal field of PathsBuffer
            num_per_src = int(np.bincount(src_indices, minlength=num_sources).max())
        else:
            num_per_src = num_stored
        self.max_fraction = num_per_src / kwargs["max_num_paths_per_src"]
        return buffer


@contextmanager
def isolate_tx(scene, tx_name: str):
    """
    Remove all other transmitters and all receivers from the scene.

    Yields a dict that maps each removed receiver name to its object, so that
    receivers can be added back in batches. On exit, the scene gets back all
    removed objects, in their original order.
    """
    removed_txs = [scene.get(name) for name in list(scene.transmitters) if name != tx_name]
    removed_rxs = [scene.get(name) for name in list(scene.receivers)]
    for obj in removed_txs + removed_rxs:
        scene.remove(obj.name)
    try:
        yield {obj.name: obj for obj in removed_rxs}
    finally:
        for name in list(scene.receivers):
            scene.remove(name)
        for obj in removed_txs + removed_rxs:
            scene.add(obj)


def iter_paths_for_receivers(
    scene,
    tx_name: str,
    rx_names: List[str],
    rx_batch_size: Optional[int] = None,
    **solver_kwargs,
) -> Iterator[Tuple[Paths, List[str], float]]:
    """
    Solve paths from one transmitter to the named receivers, in batches.

    All other transmitters and receivers are removed from the scene once, and
    each batch of receivers is added, solved and removed in turn. Sionna RT
    discards paths silently when many receivers share one solve (see
    set_specular_chain_table_size), so large receiver sets should be solved in
    batches. Receivers are selected by name, so the result does not depend on
    the order of ``scene.receivers``.

    Yields
    ------
    tuple
        (paths, row_rx_names, buffer_fill) per batch: the Paths object, the
        receiver names in the order of its receiver axis, and the fill of
        Sionna's per-source path buffer as a fraction of
        max_num_paths_per_src (see _CandidateCounter). A warning is logged
        when it reaches 1.0. The batch stays in the scene until the next item
        is requested.
    """
    missing = set(rx_names) - set(scene.receivers)
    if missing:
        raise ValueError(f"Receivers not in scene: {sorted(missing)[:5]}")
    unknown = set(solver_kwargs) - set(_SOLVER_ARGS)
    if unknown:
        raise TypeError(f"Unknown path solver arguments: {sorted(unknown)}")
    step = rx_batch_size or max(len(rx_names), 1)

    ps = PathSolver()
    counter = _CandidateCounter(ps._candidate_generator)  # internal attribute of PathSolver
    ps._candidate_generator = counter
    with isolate_tx(scene, tx_name) as receivers:
        for i in range(0, len(rx_names), step):
            batch = rx_names[i:i + step]
            for name in batch:
                scene.add(receivers[name])
            logger.info("Solving paths for %s with %s receivers", tx_name, len(batch))
            paths = ps(scene, **solver_kwargs)
            if counter.max_fraction >= 1.0:
                logger.warning(
                    "%s: the path buffer of a source is full (%.2f x max_num_paths_per_src); "
                    "paths were discarded. Reduce path_solver_rx_batch_size or raise "
                    "path_solver_max_num_paths_per_src.",
                    tx_name, counter.max_fraction,
                )
            # The receiver axis of the paths follows the scene's receiver order
            yield paths, list(scene.receivers), counter.max_fraction
            del paths  # free the GPU buffers before the next solve
            for name in batch:
                scene.remove(name)


def solve_paths_for_receivers(
    scene,
    tx_name: str,
    rx_names: List[str],
    **solver_kwargs,
) -> Tuple[Paths, List[str]]:
    """
    Solve paths from one transmitter to the named receivers in one solve.

    Large receiver sets lose paths in one solve; use iter_paths_for_receivers
    with a batch size for them.

    Returns
    -------
    tuple
        (paths, row_rx_names): the Paths object and the receiver names in the
        order of its receiver axis.
    """
    batches = iter_paths_for_receivers(
        scene, tx_name, rx_names, rx_batch_size=None, **solver_kwargs)
    if not rx_names:
        raise ValueError("rx_names is empty")
    try:
        paths, row_rx_names, _ = next(batches)
    finally:
        batches.close()  # restores the scene
    return paths, row_rx_names


def iter_paths_per_tx(
    scene,
    num_txs: int,
    num_sectors: int,
    users_per_tx: List[int],
    per_tx_users_only: bool = True,
    rx_batch_size: Optional[int] = 50,
    **solver_kwargs,
) -> Iterator[Tuple[int, Paths, List[str], float]]:
    """
    Solve paths for each TX separately, in receiver batches, one batch at a time.

    Other TXs and the receivers outside the current batch are removed from the
    scene during each solve. A Paths object holds Sionna's candidate buffer on
    the GPU (about 2 GB with 10^7 paths per source), so use each batch and
    drop it before requesting the next instead of keeping all of them.

    Parameters
    ----------
    scene : sionna.rt.Scene
        The Sionna scene object with transmitters and receivers already added
    num_txs : int
        Total number of transmitters (base stations x sectors per base station)
    num_sectors : int
        Number of sectors per base station (e.g. 1, 3, 6, etc.)
    users_per_tx : list of int
        Number of users of each TX, as returned by add_receivers_from_samples
    per_tx_users_only : bool, default=True
        If True, solve paths only for the users sampled for each TX.
        If False, solve paths from each TX to all users.
    rx_batch_size : int, optional, default=50
        Number of receivers per solve. None solves all receivers of a TX at
        once, which loses paths for large receiver sets.
    **solver_kwargs
        Arguments of Sionna's PathSolver: max_depth, max_num_paths_per_src,
        samples_per_src, synthetic_array, los, specular_reflection,
        diffuse_reflection, refraction, diffraction, edge_diffraction,
        diffraction_lit_region, seed

    Yields
    ------
    tuple
        (tx_idx, paths, row_rx_names, buffer_fill) per batch, see
        iter_paths_for_receivers
    """
    all_rx_names = [f"UE_{i}" for i in range(int(sum(users_per_tx)))]
    for tx_idx in range(num_txs):
        tx_name = f"BS_{tx_idx // num_sectors}_sector_{(tx_idx % num_sectors) + 1}"
        rx_names = rx_names_for_tx(users_per_tx, tx_idx) if per_tx_users_only else all_rx_names
        for paths, row_rx_names, buffer_fill in iter_paths_for_receivers(
                scene, tx_name, rx_names, rx_batch_size=rx_batch_size, **solver_kwargs):
            yield tx_idx, paths, row_rx_names, buffer_fill
            del paths  # free the GPU buffers before the next solve


def _paths_of(entry) -> List[Paths]:
    """The Paths objects of one TX: a Paths object, or a list of Paths objects or of (paths, ...) batches."""
    if isinstance(entry, Paths):
        return [entry]
    return [batch if isinstance(batch, Paths) else batch[0] for batch in entry]


def _valid_values_per_rx(entry, field: str) -> List[np.ndarray]:
    """For each receiver of one TX, the values of ``field`` over its valid paths (tau > 0)."""
    values = []
    for paths in _paths_of(entry):
        tau = np.asarray(paths.tau.numpy(), dtype=np.float64)
        if tau.size == 0:
            values.extend(np.zeros(0) for _ in range(tau.shape[0] if tau.ndim else 0))
            continue
        data = tau if field == "tau" else np.asarray(getattr(paths, field).numpy(), dtype=np.float64)
        for rx in range(tau.shape[0]):
            values.append(data[rx][tau[rx] > 0])
    return values


def get_doppler_stats(paths_per_tx: List) -> Tuple[List[float], List[float], List[Tuple[float, float]]]:
    """
    Get Doppler statistics per TX: mean shift, spread (std), and (min, max).

    Each entry of ``paths_per_tx`` is a Paths object, or a list of Paths
    objects or of (paths, ...) batches of one TX. Only paths with
    tau > 0 are included.
    """
    mean_doppler_shifts = []
    doppler_spreads = []
    min_max_doppler_shifts = []
    for entry in paths_per_tx:
        d = np.concatenate([np.zeros(0)] + _valid_values_per_rx(entry, "doppler"))
        if d.size == 0:
            mean_doppler_shifts.append(np.nan)
            doppler_spreads.append(np.nan)
            min_max_doppler_shifts.append((np.nan, np.nan))
        else:
            mean_doppler_shifts.append(float(np.mean(d)))
            doppler_spreads.append(float(np.std(d)))
            min_max_doppler_shifts.append((float(np.min(d)), float(np.max(d))))
    return mean_doppler_shifts, doppler_spreads, min_max_doppler_shifts


def get_delay_stats(paths_per_tx: List) -> Tuple[List[float], List[float]]:
    """
    Compute per-TX delay statistics: mean delay and RMS delay spread.

    Each entry of ``paths_per_tx`` is a Paths object, or a list of Paths
    objects or of (paths, ...) batches. Delays are taken from ``paths.tau`` (seconds), and
    only paths with tau > 0 are included.
    """
    mean_delays: List[float] = []
    rms_delay_spreads: List[float] = []
    for entry in paths_per_tx:
        d = np.concatenate([np.zeros(0)] + _valid_values_per_rx(entry, "tau"))
        if d.size == 0:
            mean_delays.append(np.nan)
            rms_delay_spreads.append(np.nan)
            continue
        mean_delay = float(np.mean(d))
        mean_delays.append(mean_delay)
        rms_delay_spreads.append(float(np.sqrt(np.mean((d - mean_delay) ** 2))))
    return mean_delays, rms_delay_spreads


def get_num_paths_histogram(
    paths_per_tx: List,
    valid_rx_mask_per_tx: Optional[List[np.ndarray]] = None,
) -> List[List[int]]:
    """
    For each TX, compute a histogram over the number of valid paths per user.

    Each entry of ``paths_per_tx`` is a Paths object, or a list of Paths
    objects or of (paths, ...) batches. If valid_rx_mask_per_tx is provided, only receivers
    with mask True are included (the mask follows the receivers of all
    batches of the TX, in order).

    Returns a list of lists, one per TX: hist[i] is the number of users of
    that TX with exactly i valid paths. Inner list lengths can differ.
    """
    path_count: List[List[int]] = []
    for tx_idx, entry in enumerate(paths_per_tx):
        counts = np.array([len(v) for v in _valid_values_per_rx(entry, "tau")], dtype=int)
        if valid_rx_mask_per_tx is not None and tx_idx < len(valid_rx_mask_per_tx):
            counts = counts[np.asarray(valid_rx_mask_per_tx[tx_idx], dtype=bool)]
        path_count.append(np.bincount(counts).tolist() if counts.size else [])
    return path_count
