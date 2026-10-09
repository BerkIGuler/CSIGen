"""
Receiver creation utilities for adding user equipment to scenes.
"""

from sionna.rt import Receiver
import numpy as np
from typing import List, Tuple
import logging

from src.user_equipment import generate_ue_parameters
from src.utils import get_tx_color

logger = logging.getLogger(__name__)


def rx_names_for_tx(users_per_tx: List[int], tx_idx: int) -> List[str]:
    """
    Names of the users sampled for TX ``tx_idx``.

    Users are named UE_<k> with k counting TX by TX (see
    add_receivers_from_samples), so the users of TX t are
    UE_<sum(users_per_tx[:t])> to UE_<sum(users_per_tx[:t+1]) - 1>.
    """
    start = int(sum(users_per_tx[:tx_idx]))
    return [f"UE_{k}" for k in range(start, start + int(users_per_tx[tx_idx]))]


def add_receivers_from_samples(
    scene,
    sampled_positions: Tuple[List[np.ndarray], List[np.ndarray]],
    num_sectors: int,
    mobility_preset: str,
    mobility_presets: dict,
    seed: int = 1
) -> Tuple[int, List[int], int]:
    """
    Add receivers to scene from sampled positions.
    
    Parameters
    ----------
    scene : sionna.rt.Scene
        The Sionna scene object
    sampled_positions : tuple
        (positions, cell_ids) from sample_user_positions(): lists with one
        entry per TX, of shapes [num_users_tx, 3] and [num_users_tx]
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
        (num_txs, users_per_tx, total_users), where users_per_tx lists the
        number of users of each TX. Users are named UE_<k> with k counting
        TX by TX; rx_names_for_tx() gives the names of the users of a TX.
    """
    # Extract positions and cell_ids
    positions, _ = sampled_positions
    num_txs = len(positions)
    users_per_tx = [len(p) for p in positions]
    total_users = sum(users_per_tx)
    if total_users == 0:
        logger.warning(f"Total TXs: {num_txs}, no users to add")
    else:
        logger.info(f"Total TXs: {num_txs}, users per TX: {min(users_per_tx)}-{max(users_per_tx)}, total users: {total_users}")
    
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
        
        for user_idx in range(users_per_tx[tx_idx]):
            pos = positions[tx_idx][user_idx].tolist()  # [x, y, z]
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
    
    return num_txs, users_per_tx, total_users
