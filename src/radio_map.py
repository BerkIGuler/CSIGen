"""
Radio map solving and user position sampling utilities.
"""

from sionna.rt import RadioMapSolver
from typing import Tuple, Optional
import numpy as np
import mitsuba as mi


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
    samples_per_tx: int = 10**8
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
        'samples_per_tx': samples_per_tx
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
