"""
Receiver creation utilities for adding user equipment to scenes.
"""

from sionna.rt import Receiver
import numpy as np
from typing import Tuple
import logging

from src.user_equipment import generate_ue_parameters
from src.utils import get_tx_color

logger = logging.getLogger(__name__)


def add_receivers_from_samples(
    scene,
    sampled_positions: Tuple[np.ndarray, np.ndarray],
    num_sectors: int,
    mobility_preset: str,
    mobility_presets: dict,
    seed: int = 1
) -> Tuple[int, int, int]:
    """
    Add receivers to scene from sampled positions.
    
    Parameters
    ----------
    scene : sionna.rt.Scene
        The Sionna scene object
    sampled_positions : tuple
        (positions, cell_ids) from sample_user_positions()
        - positions: Tensor/array with shape [num_tx, num_users_per_tx, 3]
        - cell_ids: Tensor/array with shape [num_tx, num_users_per_tx]
    num_sectors : int
        Number of sectors per base station
    mobility_preset : str
        Key for mobility preset in mobility_presets dict
    mobility_presets : dict
        Dictionary of mobility presets
    seed : int, default=1
        Seed for UE parameter generation
    
    Returns
    -------
    tuple
        (num_txs, num_users_per_tx, total_users)
    """
    # Extract positions and cell_ids
    positions_tensor, cell_ids_tensor = sampled_positions
    
    # Convert to numpy once for efficient iteration
    positions = positions_tensor.numpy() if hasattr(positions_tensor, 'numpy') else np.array(positions_tensor)
    
    num_txs, num_users_per_tx, _ = positions.shape
    total_users = num_txs * num_users_per_tx
    logger.info(f"Total TXs: {num_txs}, Users per TX: {num_users_per_tx}, Total users: {total_users}")
    
    # Get config from selected preset and generate UE parameters
    if mobility_preset not in mobility_presets:
        raise ValueError(f"Mobility preset '{mobility_preset}' not found in mobility_presets")
    preset_config = mobility_presets[mobility_preset]
    orientation_mode = preset_config.get("orientation_mode", "random")
    
    orientations, velocities = generate_ue_parameters(
        num_ues=total_users,
        seed=seed,
        **preset_config
    )
    
    # Add receivers for each user
    user_count = 0
    for tx_idx in range(num_txs):
        # Map tx_idx to TX name: BS_{bs_id}_sector_{sector_id}
        bs_id = tx_idx // num_sectors
        sector_id = (tx_idx % num_sectors) + 1
        tx_name = f"BS_{bs_id}_sector_{sector_id}"
        
        # Get the TX object from the scene (needed for look_at mode)
        tx_object = scene.get(tx_name)
        
        # Color for this TX's users (for visualization purposes)
        color = get_tx_color(tx_idx, num_txs)
        
        for user_idx in range(num_users_per_tx):
            pos = positions[tx_idx, user_idx].tolist()  # [x, y, z]
            vel = velocities[user_count].tolist()       # [vx, vy, vz]
            
            if orientation_mode == "random":
                rx = Receiver(
                    name=f"UE_{user_count}",
                    position=pos,
                    orientation=orientations[user_count].tolist(),
                    velocity=vel,
                    color=color
                )
            else:  # "to_tx"
                rx = Receiver(
                    name=f"UE_{user_count}",
                    position=pos,
                    look_at=tx_object,
                    velocity=vel,
                    color=color
                )
            
            scene.add(rx)
            user_count += 1
    
    logger.info(f"Added {user_count} receivers to scene")
    logger.info(f"  - Preset: {mobility_preset}")
    logger.info(f"  - Orientation: {orientation_mode}, Speed: {preset_config.get('speed_distribution')}")
    
    return num_txs, num_users_per_tx, total_users
