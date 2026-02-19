"""
Radio map solving and user position sampling utilities.
"""

from sionna.rt import RadioMapSolver
from typing import Tuple, Optional
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample user positions from radio map.
    
    Parameters
    ----------
    radio_map : RadioMap
        RadioMap object from solve_radio_map()
    num_pos_per_tx : int
        Number of user samples to generate per TX
    metric : str, default="path_gain"
        Metric for the user sampling from ["path_gain", "rss", "sinr"].
    min_val_db : float, default=-150
        Minimum value in dB for the user sampling
    max_val_db : float, optional
        Maximum value in dB for the user sampling. If None, no upper bound is applied.
    min_dist : float, default=0.0
        Minimum distance in meters from TX for the user sampling
    max_dist : float, optional
        Maximum distance in meters from TX for the user sampling. If None, no upper bound is applied.
    tx_association : bool, default=True
        Whether to use TX association for the user sampling
    center_pos : bool, default=True
        Whether to sample from the radio map cell center. If False, the user is sampled at a random position within the cell.
    seed : int, default=1
        Seed for the user sampling
    
    Returns
    -------
    tuple
        (positions, cell_ids)
        - positions: Tensor with shape [num_tx, num_users_per_tx, 3]
        - cell_ids: Tensor with shape [num_tx, num_users_per_tx]
    """
    sample_kwargs = {
        "num_pos": num_pos_per_tx,
        "metric": metric,
        "min_val_db": min_val_db,
        "tx_association": tx_association,
        "center_pos": center_pos,
        "seed": seed,
    }
    if max_val_db is not None:
        sample_kwargs["max_val_db"] = max_val_db
    if min_dist is not None:
        sample_kwargs["min_dist"] = min_dist
    if max_dist is not None:
        sample_kwargs["max_dist"] = max_dist
    sampled_positions = radio_map.sample_positions(**sample_kwargs)
    return sampled_positions[0], sampled_positions[1]  # positions, cell_ids


def filter_positions_by_edge_distance(
    sampled_positions: Tuple[np.ndarray, np.ndarray],
    edge_epsilon: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter sampled positions to remove users within a specified distance from measurement surface edges.
    
    This function computes the actual bounds of the measurement surface from the sampled positions
    and filters out positions that are too close to the edges. This is useful for avoiding edge
    effects in channel simulations.
    
    Parameters
    ----------
    sampled_positions : tuple
        Tuple of (positions, cell_ids) where:
        - positions: Tensor/array with shape [num_tx, num_users_per_tx, 3]
        - cell_ids: Tensor/array with shape [num_tx, num_users_per_tx]
    edge_epsilon : float
        Minimum distance in meters from measurement surface edges to keep users.
        Users within this distance from any edge will be filtered out.
        If 0.0 or negative, no filtering is performed.
    
    Returns
    -------
    tuple
        (filtered_positions, filtered_cell_ids) with shape [num_tx, min_users_per_tx, 3] and
        [num_tx, min_users_per_tx] respectively. All TXs will have the same number of users
        (the minimum across all TXs after filtering).
    
    Notes
    -----
    - The bounds are computed from the actual sampled positions, which represent the
      measurement surface bounds more accurately than scene bounds (when
      the measurement surface is clipped to building bounds).
    - After filtering, all TXs are truncated to have the same number of users to maintain
      consistent array shapes. This uses the minimum number of users across all TXs.
    """
    if edge_epsilon <= 0.0:
        logger.info(f"edge_epsilon is {edge_epsilon}, skipping edge filtering")
        return sampled_positions
    
    # Convert to numpy for filtering
    positions_np = sampled_positions[0]
    cell_ids_np = sampled_positions[1]
    
    # Handle tensorflow tensors
    if hasattr(positions_np, 'numpy'):
        positions_np = positions_np.numpy()
    if hasattr(cell_ids_np, 'numpy'):
        cell_ids_np = cell_ids_np.numpy()
    
    num_txs, num_users_per_tx, _ = positions_np.shape
    
    # Compute actual bounds from sampled positions (measurement surface bounds)
    # This is more accurate than scene bounds since the radio map may be computed
    # on a clipped measurement surface
    all_x_coords = positions_np[:, :, 0].flatten()
    all_y_coords = positions_np[:, :, 1].flatten()
    x_min = float(np.min(all_x_coords))
    x_max = float(np.max(all_x_coords))
    y_min = float(np.min(all_y_coords))
    y_max = float(np.max(all_y_coords))
    
    logger.info(f"Measurement surface bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
    logger.info(f"Filtering users within {edge_epsilon:.1f} m of measurement surface edges...")
    
    # Filter positions for each TX
    filtered_positions = []
    filtered_cell_ids = []
    
    total_before = 0
    total_after = 0
    
    for tx_idx in range(num_txs):
        # Get positions for this TX
        tx_positions = positions_np[tx_idx]  # [num_users_per_tx, 3]
        tx_cell_ids = cell_ids_np[tx_idx]    # [num_users_per_tx]
        
        # Check distance from edges for each user
        x_coords = tx_positions[:, 0]
        y_coords = tx_positions[:, 1]
        
        # Keep users that are at least epsilon away from all edges
        keep_mask = (
            (x_coords >= x_min + edge_epsilon) & 
            (x_coords <= x_max - edge_epsilon) &
            (y_coords >= y_min + edge_epsilon) & 
            (y_coords <= y_max - edge_epsilon)
        )
        
        # Filter positions and cell_ids
        filtered_tx_positions = tx_positions[keep_mask]
        filtered_tx_cell_ids = tx_cell_ids[keep_mask]
        
        filtered_positions.append(filtered_tx_positions)
        filtered_cell_ids.append(filtered_tx_cell_ids)
        
        num_kept = np.sum(keep_mask)
        num_removed = len(keep_mask) - num_kept
        total_before += len(keep_mask)
        total_after += num_kept
        
        logger.info(f"  TX {tx_idx}: kept {num_kept}/{len(keep_mask)} users (removed {num_removed})")
    
    # Find the minimum number of users per TX to maintain consistent shape
    min_users_per_tx = min(len(fp) for fp in filtered_positions)
    
    if min_users_per_tx == 0:
        logger.warning("All users were filtered out for at least one TX!")
        logger.warning("Consider reducing edge_epsilon or increasing num_user_samples_per_tx")
        # Return empty arrays with correct shape
        filtered_positions_array = np.empty((num_txs, 0, 3))
        filtered_cell_ids_array = np.empty((num_txs, 0), dtype=cell_ids_np.dtype)
        return filtered_positions_array, filtered_cell_ids_array
    
    # Truncate all TXs to have the same number of users (use first min_users_per_tx)
    for tx_idx in range(num_txs):
        filtered_positions[tx_idx] = filtered_positions[tx_idx][:min_users_per_tx]
        filtered_cell_ids[tx_idx] = filtered_cell_ids[tx_idx][:min_users_per_tx]
    
    # Stack into arrays
    filtered_positions_array = np.stack(filtered_positions, axis=0)  # [num_tx, min_users_per_tx, 3]
    filtered_cell_ids_array = np.stack(filtered_cell_ids, axis=0)   # [num_tx, min_users_per_tx]
    
    logger.info(f"Filtering complete: {total_before} -> {total_after} users (removed {total_before - total_after})")
    logger.info(f"Final users per TX: {min_users_per_tx}")
    
    return filtered_positions_array, filtered_cell_ids_array
