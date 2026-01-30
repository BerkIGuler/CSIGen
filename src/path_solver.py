"""
Path solving utilities for efficient per-TX path computation.
"""

from sionna.rt import PathSolver
from typing import List
import logging

logger = logging.getLogger(__name__)


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
    # PathSolver is computationally expensive, so we solve for each TX separately
    ps = PathSolver()
    
    # Get all receivers and TXs from scene
    all_rx_names = [obj for obj in scene.receivers]
    all_tx_names = [obj for obj in scene.transmitters]
    
    paths_per_tx = []
    
    for tx_idx in range(num_txs):
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
            logger.info(f"TX {tx_idx} ({tx_name}): Solving paths for {len(selected_rx_names)} associated users (UE_{start_idx} to UE_{end_idx-1})")
        else:
            # Use all receivers
            selected_rx_names = all_rx_names
            logger.info(f"TX {tx_idx} ({tx_name}): Solving paths for all {len(selected_rx_names)} users")
        
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
            logger.info(f"Solving paths for TX {tx_idx} ({tx_name}) with {len(selected_rx_names)} receivers")
            logger.info(f"Scene receivers: {len(scene.receivers)}")
            logger.info(f"Scene transmitters: {len(scene.transmitters)}")
            
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
                seed=seed
            )
            paths_per_tx.append(paths_tx)
            
        finally:
            # Restore removed receivers and TXs
            for rx_obj in removed_rxs:
                scene.add(rx_obj)
            for tx_obj in removed_txs:
                scene.add(tx_obj)
    
    logger.info(f"\nCompleted path solving for {len(paths_per_tx)} TXs")
    return paths_per_tx
