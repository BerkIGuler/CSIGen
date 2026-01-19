"""
Radio map solving and user position sampling utilities.
"""

from sionna.rt import RadioMapSolver
from typing import Tuple
import numpy as np


def solve_radio_map(
    scene,
    measurement_surface=None,
    diffuse_reflection: bool = True,
    diffraction: bool = True,
    edge_diffraction: bool = True,
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
        Measurement surface mesh (if None, uses scene's measurement surface)
    diffuse_reflection : bool, default=True
        Whether to include diffuse reflection
    diffraction : bool, default=True
        Whether to include diffraction
    edge_diffraction : bool, default=True
        Whether to include edge diffraction
    max_depth : int, default=5
        Maximum number of ray scene interactions
    samples_per_tx : int, default=10**8
        Number of samples per TX antenna array
    
    Returns
    -------
    RadioMap
        RadioMap object (MeshRadioMap or PlanarRadioMap)
    """
    rm_solver = RadioMapSolver()
    rm = rm_solver(
        scene,
        measurement_surface=measurement_surface,
        diffuse_reflection=diffuse_reflection,
        diffraction=diffraction,
        edge_diffraction=edge_diffraction,
        max_depth=max_depth,
        samples_per_tx=samples_per_tx
    )
    return rm


def sample_user_positions(
    radio_map,
    num_pos_per_tx: int,
    metric: str = "path_gain",
    min_val_db: float = -150,
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
        Metric for the user sampling
    min_val_db : float, default=-150
        Minimum value in dB for the user sampling
    tx_association : bool, default=True
        Whether to use TX association for the user sampling
    center_pos : bool, default=True
        Whether to sample from the radio map cell center
    seed : int, default=1
        Seed for the user sampling
    
    Returns
    -------
    tuple
        (positions, cell_ids)
        - positions: Tensor with shape [num_tx, num_users_per_tx, 3]
        - cell_ids: Tensor with shape [num_tx, num_users_per_tx]
    """
    sampled_positions = radio_map.sample_positions(
        num_pos=num_pos_per_tx,
        metric=metric,
        min_val_db=min_val_db,
        tx_association=tx_association,
        center_pos=center_pos,
        seed=seed
    )
    return sampled_positions[0], sampled_positions[1]  # positions, cell_ids
