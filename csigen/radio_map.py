"""
Radio map solving and user position sampling utilities.
"""

from sionna.rt import RadioMapSolver
from typing import List, Tuple, Optional
import numpy as np
import mitsuba as mi
import logging

logger = logging.getLogger(__name__)


def solve_radio_map(
    scene,
    measurement_surface=None,
    center: Optional[mi.Point3f] = None,
    orientation: Optional[mi.Point3f] = None,
    size: Optional[mi.Point2f] = None,
    cell_size: Optional[mi.Point2f] = None,
    specular_reflection: bool = True,
    diffuse_reflection: bool = True,
    refraction: bool = True,
    diffraction: bool = True,
    edge_diffraction: bool = True,
    diffraction_lit_region: bool = False,
    max_depth: int = 5,
    samples_per_tx: int = 10**8,
    seed: int = 1
):
    """
    Solve radio map for the scene.
    
    Parameters
    ----------
    scene : sionna.rt.Scene
        The Sionna scene object
    measurement_surface : mesh, optional
        Measurement surface mesh. If set, the radio map is computed for this surface,
        where every triangle in the mesh is a cell in the radio map.
        If None, the radio map is computed for a measurement grid defined by
        center, orientation, size, and cell_size.
    center : mi.Point3f, optional
        Center of the radio map measurement plane [m] as a three-dimensional vector.
        Ignored if measurement_surface is provided.
        If None, the radio map is centered on the center of the scene at 1.5m elevation.
        If not None, orientation and size must be provided.
    orientation : mi.Point3f, optional
        Orientation of the radio map measurement plane specified through three angles
        corresponding to a 3D rotation. Ignored if measurement_surface is provided.
        An orientation of None corresponds to a radio map parallel to the XY plane.
        If not None, center and size must be provided.
    size : mi.Point2f, optional
        Size of the radio map measurement plane [m]. Ignored if measurement_surface is provided.
        If None, the size covers the entire scene. If not None, center and orientation must be provided.
    cell_size : mi.Point2f, optional
        Size of a cell of the radio map measurement plane [m].
        Ignored if measurement_surface is provided.
        Required if measurement_surface is None.
    specular_reflection : bool, default=True
        Whether to include specular reflection
    diffuse_reflection : bool, default=True
        Whether to include diffuse reflection
    refraction : bool, default=True
        Whether to include refraction
    diffraction : bool, default=True
        Whether to include diffraction
    edge_diffraction : bool, default=True
        Whether to include edge diffraction
    diffraction_lit_region : bool, default=False
        Whether to include diffraction in the lit region
    max_depth : int, default=5
        Maximum number of ray scene interactions
    samples_per_tx : int, default=10**8
        Number of samples per TX antenna array
    seed : int, default=1
        Random seed for the radio map solver (reproducibility)
    
    Returns
    -------
    RadioMap
        RadioMap object (MeshRadioMap if measuremenet surface is provided, or PlanarRadioMap if it is not provided)
    
    Raises
    ------
    ValueError
        If measurement_surface is None and required parameters are not provided
    """
    # Validate parameters when measurement_surface is None
    if measurement_surface is None:
        # cell_size is required when measurement_surface is None
        if cell_size is None:
            raise ValueError(
                "cell_size must be provided when measurement_surface is None. "
                "The radio map requires cell_size to define the measurement grid."
            )
        
        # Validate interdependent parameters
        if center is not None:
            if orientation is None or size is None:
                raise ValueError(
                    "When center is provided, both orientation and size must be provided. "
                    "If center is None, the radio map will be centered on the scene center at 1.5m elevation."
                )
        
        if orientation is not None:
            if center is None or size is None:
                raise ValueError(
                    "When orientation is provided, both center and size must be provided. "
                    "If orientation is None, the radio map will be parallel to the XY plane."
                )
        
        if size is not None:
            if center is None or orientation is None:
                raise ValueError(
                    "When size is provided, both center and orientation must be provided. "
                    "If size is None, the radio map will cover the entire scene."
                )
    
    rm_solver = RadioMapSolver()
    
    # Build kwargs for RadioMapSolver
    solver_kwargs = {
        'measurement_surface': measurement_surface,
        'specular_reflection': specular_reflection,
        'diffuse_reflection': diffuse_reflection,
        'refraction': refraction,
        'diffraction': diffraction,
        'edge_diffraction': edge_diffraction,
        'diffraction_lit_region': diffraction_lit_region,
        'max_depth': max_depth,
        'samples_per_tx': samples_per_tx,
        'seed': seed,
    }
    
    # Add grid parameters only if measurement_surface is None
    if measurement_surface is None:
        if center is not None:
            solver_kwargs['center'] = center
        if orientation is not None:
            solver_kwargs['orientation'] = orientation
        if size is not None:
            solver_kwargs['size'] = size
        solver_kwargs['cell_size'] = cell_size
    
    rm = rm_solver(scene, **solver_kwargs)
    return rm


def sample_user_positions(
    radio_map,
    num_pos_per_tx: int,
    metric: str = "path_gain",
    min_val_db: float = -150,
    max_val_db: Optional[float] = None,
    min_dist: float = 0.0,
    max_dist: Optional[float] = None,
    tx_association: bool = True,
    center_pos: bool = True,
    seed: int = 1
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Sample user positions from radio map, in distinct cells.

    A cell is valid for a TX if its metric lies in [min_val_db, max_val_db],
    its center lies within [min_dist, max_dist] of the TX and, with
    ``tx_association``, the TX has the highest metric in the cell (the same
    rules as Sionna's ``RadioMap.sample_cells``). Unlike Sionna, which draws
    cells with replacement, each TX gets ``min(num_pos_per_tx, number of valid
    cells)`` distinct cells, so no two users of a TX share a cell. A TX with
    no valid cell gets no users.

    Parameters
    ----------
    radio_map : RadioMap
        RadioMap object from solve_radio_map()
    num_pos_per_tx : int
        Maximum number of users sampled per TX
    metric : str, default="path_gain"
        Metric for the user sampling from ["path_gain", "rss", "sinr"].
    min_val_db : float, default=-150
        Minimum value in dB (dBm for "rss") for the user sampling
    max_val_db : float, optional
        Maximum value in dB for the user sampling. If None, no upper bound is applied.
    min_dist : float, default=0.0
        Minimum distance in meters from TX for the user sampling
    max_dist : float, optional
        Maximum distance in meters from TX for the user sampling. If None, no upper bound is applied.
    tx_association : bool, default=True
        Only sample cells in which the TX has the highest metric
    center_pos : bool, default=True
        Place users at the cell centers. If False, users are placed uniformly at random
        within their cell (a triangle of the measurement surface).
    seed : int, default=1
        Seed for the user sampling

    Returns
    -------
    tuple
        (positions, cell_ids): lists with one entry per TX, of shapes
        [num_users_tx, 3] and [num_users_tx]; num_users_tx can differ between TXs.
    """
    if metric not in ("path_gain", "rss", "sinr"):
        raise ValueError(f"Invalid metric: {metric}")
    num_tx = radio_map.num_tx
    values = np.array(getattr(radio_map, metric)).reshape(num_tx, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "rss":
            values_db = 10.0 * np.log10(values) + 30.0  # W to dBm
        else:
            values_db = 10.0 * np.log10(values)
    hi = np.inf if max_val_db is None else max_val_db
    # NaN and -inf values fail these comparisons and are never valid
    valid = (values_db >= min_val_db) & (values_db <= hi)

    cc = radio_map.cell_centers  # mi.Point3f; np.array() of it is [3, num_cells]
    centers = np.stack([np.array(cc.x), np.array(cc.y), np.array(cc.z)], axis=1)
    # Transmitter positions as stored by Sionna (internal attribute of RadioMap)
    tp = radio_map._tx_positions
    tx_pos = np.stack([np.array(tp.x), np.array(tp.y), np.array(tp.z)], axis=1)
    dist = np.linalg.norm(centers[None, :, :] - tx_pos[:, None, :], axis=2)
    valid &= (dist >= (min_dist or 0.0)) & (dist <= (np.inf if max_dist is None else max_dist))
    if tx_association:
        best = np.array(radio_map.tx_association(metric)).reshape(-1)
        valid &= best[None, :] == np.arange(num_tx)[:, None]

    if not center_pos:
        mesh = radio_map.measurement_surface
        faces = np.array(mesh.faces_buffer()).reshape(-1, 3)
        vertices = np.array(mesh.vertex_positions_buffer()).reshape(-1, 3)

    rng = np.random.default_rng(seed)
    pos_rng = np.random.default_rng([seed, 1])  # positions within cells
    positions, cell_ids = [], []
    for tx in range(num_tx):
        # Each cell gets a random key that depends only on the seed, and the TX
        # takes its valid cells with the lowest keys. A cell that changes
        # validity (the radio map is not bit-identical between GPU runs)
        # changes at most one user instead of the whole draw.
        keys = rng.random(valid.shape[1])
        # Position of the user within each cell, also fixed per cell
        offsets = pos_rng.random((valid.shape[1], 2)) if not center_pos else None
        candidates = np.flatnonzero(valid[tx])
        n = min(num_pos_per_tx, len(candidates))
        cells = candidates[np.argsort(keys[candidates], kind="stable")[:n]]
        if center_pos:
            pos = centers[cells]
        else:
            v0, v1, v2 = (vertices[faces[cells, k]] for k in range(3))
            r = offsets[cells]
            a = np.sqrt(r[:, :1])
            pos = (1 - a) * v0 + a * (1 - r[:, 1:]) * v1 + a * r[:, 1:] * v2
        positions.append(pos.astype(np.float32))
        cell_ids.append(cells)
        if n < num_pos_per_tx:
            logger.info("TX %s: %s valid cells, sampled %s of %s users", tx, len(candidates), n, num_pos_per_tx)

    counts = [len(c) for c in cell_ids]
    if sum(counts) == 0:
        logger.warning("No valid cell for any TX, no users sampled")
    else:
        logger.info("Sampled %s users in distinct cells (per TX: min %s, max %s)", sum(counts), min(counts), max(counts))
    return positions, cell_ids


def filter_positions_by_edge_distance(
    sampled_positions: Tuple[List[np.ndarray], List[np.ndarray]],
    edge_epsilon: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Remove users within ``edge_epsilon`` meters of the edges of the sampled area.

    A heuristic: users at the far edges of the (clipped) measurement surface
    tend to be LoS-dominated and are removed. The bounds are taken from the
    sampled positions of all TXs. Each TX keeps its own remaining users, so
    the number of users per TX can differ.

    Parameters
    ----------
    sampled_positions : tuple
        (positions, cell_ids) from sample_user_positions(): lists with one
        entry per TX, of shapes [num_users_tx, 3] and [num_users_tx]
    edge_epsilon : float
        Minimum distance in meters from the edges. If 0.0 or negative, no
        filtering is performed.

    Returns
    -------
    tuple
        (positions, cell_ids) in the same format, restricted to the kept users.
    """
    if edge_epsilon <= 0.0:
        logger.info(f"edge_epsilon is {edge_epsilon}, skipping edge filtering")
        return sampled_positions

    positions, cell_ids = sampled_positions
    if not any(len(p) for p in positions):
        logger.warning("No sampled users, skipping edge filtering")
        return sampled_positions
    all_pos = np.concatenate([p for p in positions if len(p)], axis=0)
    x_min, y_min = all_pos[:, 0].min(), all_pos[:, 1].min()
    x_max, y_max = all_pos[:, 0].max(), all_pos[:, 1].max()
    logger.info(f"Sampled area bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")

    kept_positions, kept_cell_ids = [], []
    for tx_idx, (pos, cells) in enumerate(zip(positions, cell_ids)):
        keep = (
            (pos[:, 0] >= x_min + edge_epsilon) & (pos[:, 0] <= x_max - edge_epsilon) &
            (pos[:, 1] >= y_min + edge_epsilon) & (pos[:, 1] <= y_max - edge_epsilon)
        )
        kept_positions.append(pos[keep])
        kept_cell_ids.append(cells[keep])
        if (~keep).any():
            logger.info(f"  TX {tx_idx}: kept {int(keep.sum())}/{len(keep)} users")

    before = sum(len(p) for p in positions); after = sum(len(p) for p in kept_positions)
    logger.info(f"Edge filtering: {before} -> {after} users")
    return kept_positions, kept_cell_ids
