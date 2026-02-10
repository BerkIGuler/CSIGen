"""
Main API for channel generation from configuration.

This module provides the primary interface for generating channel data
using Sionna RT based on a configuration dictionary.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, Any

from src.scene_setup import setup_scene
from src.base_station import set_tx_antenna_array, add_base_station
from src.user_equipment import set_rx_antenna_array
from src.radio_map import solve_radio_map, sample_user_positions, filter_positions_by_edge_distance
from src.receivers import add_receivers_from_samples
from src.path_solver import _solve_paths_for_single_tx
from src.channel import compute_cfr_for_paths

logger = logging.getLogger(__name__)

def generate_channels(config: Dict) -> Iterator[Dict[str, Any]]:
    """
    Main function to generate channels from config.
    
    This is the primary API for the software package.
    Takes a config dictionary and returns channel data + metadata.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with all parameters. Required keys:
        
        Scene:
        - scene_xml_path (Path or str): Path to scene XML file
        - carrier_frequency (float): Carrier frequency in Hz
        
        Antenna Arrays:
        - tx_num_rows, tx_num_cols, tx_vertical_spacing, tx_horizontal_spacing
        - tx_pattern, tx_polarization
        - rx_num_rows, rx_num_cols, rx_vertical_spacing, rx_horizontal_spacing
        - rx_pattern, rx_polarization
        
        Base Stations:
        - num_sectors, mechanical_tilt, azimuth_offset
        - tx_power_dbm
        
        Radio Map:
        - radio_map_diffuse_reflection, radio_map_diffraction, radio_map_edge_diffraction
        - radio_map_max_depth, radio_map_samples_per_tx
        
        User Sampling:
        - num_user_samples_per_tx, user_sample_seed, user_sample_min_val_db
        - user_sample_metric, tx_association, sample_center_pos
        
        Mobility:
        - mobility_preset (str): Key for mobility preset
        - mobility_presets (dict): Dictionary of mobility presets
        
        Path Solver:
        - path_solver_max_depth, path_solver_max_num_paths_per_src, path_solver_samples_per_src
        - path_solver_synthetic_array, path_solver_los_mode
        - path_solver_specular_reflection, path_solver_diffuse_reflection
        - path_solver_refraction, path_solver_diffraction, path_solver_edge_diffraction
        - path_solver_diffraction_lit_region, path_solver_seed
        - path_solver_per_tx_users_only
        
        OFDM/CFR:
        - num_subcarriers, num_ofdm_symbols, subcarrier_spacing
        - cfr_normalize_delays, cfr_normalize, cfr_out_type
        
        Scene Setup:
        - scene_center, antenna_height_offset, num_deployment_buildings
        - clip_terrain_to_buildings, terrain_clip_margin, user_shift_from_ground
        
    Yields
    ------
    dict
        Per-TX dictionary with keys:
        - 'tx_idx': int
        - 'tx_name': str
        - 'h_tx': np.ndarray (CFR for this TX)
        - 'tx_metadata': dict with TX/RX positions and global config-derived info
    """
    # Extract config parameters
    scene_xml_path = Path(config['scene_xml_path'])
    carrier_frequency = config['carrier_frequency']
    
    # Scene setup parameters
    scene_center = config['scene_center']
    antenna_height_offset = config['antenna_height_offset']
    num_deployment_buildings = config['num_deployment_buildings']
    clip_terrain = config['clip_terrain_to_buildings']
    terrain_clip_margin = config['terrain_clip_margin']
    user_shift_from_ground = config['user_shift_from_ground']
    
    # Step 1: Setup scene
    logger.info("Step 1: Setting up scene...")
    scene, building_positions, measurement_surface, antenna_information = setup_scene(
        scene_xml_path=scene_xml_path,
        carrier_frequency=carrier_frequency,
        scene_center=scene_center,
        antenna_height_offset=antenna_height_offset,
        num_deployment_buildings=num_deployment_buildings,
        clip_terrain=clip_terrain,
        terrain_clip_margin=terrain_clip_margin,
        user_shift_from_ground=user_shift_from_ground
    )
    
    # Step 2: Set antenna arrays
    logger.info("Step 2: Setting antenna arrays...")
    set_tx_antenna_array(
        scene,
        num_rows=config['tx_num_rows'],
        num_cols=config['tx_num_cols'],
        vertical_spacing=config['tx_vertical_spacing'],
        horizontal_spacing=config['tx_horizontal_spacing'],
        pattern=config['tx_pattern'],
        polarization=config['tx_polarization']
    )
    
    set_rx_antenna_array(
        scene,
        num_rows=config['rx_num_rows'],
        num_cols=config['rx_num_cols'],
        vertical_spacing=config['rx_vertical_spacing'],
        horizontal_spacing=config['rx_horizontal_spacing'],
        pattern=config['rx_pattern'],
        polarization=config['rx_polarization']
    )
    
    # Step 3: Add base stations
    logger.info("Step 3: Adding base stations...")
    num_sectors = config['num_sectors']
    for i, (building_id, antenna_position) in enumerate(antenna_information):
        bs_name = f"BS_{i}"
        add_base_station(
            scene,
            bs_name,
            position=antenna_position,
            num_sectors=num_sectors,
            mechanical_tilt=config['mechanical_tilt'],
            azimuth_offset=config['azimuth_offset'],
            tx_power_dbm=config['tx_power_dbm']
        )
    
    # Step 4: Solve radio map
    logger.info("Step 4: Solving radio map...")
    radio_map = solve_radio_map(
        scene,
        measurement_surface=measurement_surface,
        specular_reflection=config['radio_map_specular_reflection'],
        diffuse_reflection=config['radio_map_diffuse_reflection'],
        refraction=config['radio_map_refraction'],
        diffraction=config['radio_map_diffraction'],
        edge_diffraction=config['radio_map_edge_diffraction'],
        diffraction_lit_region=config['radio_map_diffraction_lit_region'],
        max_depth=config['radio_map_max_depth'],
        samples_per_tx=config['radio_map_samples_per_tx']
    )
    
    # Step 5: Sample user positions
    logger.info("Step 5: Sampling user positions...")
    sampled_positions = sample_user_positions(
        radio_map,
        num_pos_per_tx=config['num_user_samples_per_tx'],
        metric=config['user_sample_metric'],
        min_val_db=config['user_sample_min_val_db'],
        max_val_db=config['user_sample_max_val_db'],
        min_dist=config['user_sample_min_dist'],
        max_dist=config['user_sample_max_dist'],
        tx_association=config['tx_association'],
        center_pos=config['sample_center_pos'],
        seed=config['user_sample_seed']
    )
    
    # Step 5.5: Filter positions by edge distance
    if config.get('scene_edge_epsilon', 0.0) > 0.0:
        logger.info("Step 5.5: Filtering positions by edge distance...")
        sampled_positions = filter_positions_by_edge_distance(
            sampled_positions,
            edge_epsilon=config['scene_edge_epsilon']
        )
    
    # Step 6: Add receivers
    logger.info("Step 6: Adding receivers...")
    num_txs_actual, num_users_per_tx, total_users = add_receivers_from_samples(
        scene,
        sampled_positions,
        num_sectors=num_sectors,
        mobility_preset=config['mobility_preset'],
        mobility_presets=config['mobility_presets'],
        seed=config['user_sample_seed']
    )
    
    # Step 7 & 8: Solve paths and compute CFR per TX in a streaming fashion
    logger.info("Step 7 & 8: Solving paths and computing CFR per TX (streaming)...")

    per_tx_users_only = config['path_solver_per_tx_users_only']

    for tx_idx in range(num_txs_actual):
        # Solve paths for this TX only
        paths_tx = _solve_paths_for_single_tx(
            scene=scene,
            tx_idx=tx_idx,
            num_txs=num_txs_actual,
            num_sectors=num_sectors,
            num_users_per_tx=num_users_per_tx,
            per_tx_users_only=per_tx_users_only,
            max_depth=config['path_solver_max_depth'],
            max_num_paths_per_src=config['path_solver_max_num_paths_per_src'],
            samples_per_src=config['path_solver_samples_per_src'],
            synthetic_array=config['path_solver_synthetic_array'],
            los=config['path_solver_los_mode'],
            specular_reflection=config['path_solver_specular_reflection'],
            diffuse_reflection=config['path_solver_diffuse_reflection'],
            refraction=config['path_solver_refraction'],
            diffraction=config['path_solver_diffraction'],
            edge_diffraction=config['path_solver_edge_diffraction'],
            diffraction_lit_region=config['path_solver_diffraction_lit_region'],
            seed=config['path_solver_seed'],
        )

        # Compute CFR for this TX only
        h_tx = compute_cfr_for_paths(
            paths_tx=paths_tx,
            num_subcarriers=config['num_subcarriers'],
            num_ofdm_symbols=config['num_ofdm_symbols'],
            subcarrier_spacing=config['subcarrier_spacing'],
            normalize_delays=config['cfr_normalize_delays'],
            normalize=config['cfr_normalize'],
            out_type=config['cfr_out_type'],
        )

        # Derive TX name and position
        bs_id = tx_idx // num_sectors
        sector_id = (tx_idx % num_sectors) + 1
        tx_name = f"BS_{bs_id}_sector_{sector_id}"
        tx_obj = scene.get(tx_name)
        tx_pos = tx_obj.position
        if hasattr(tx_pos, 'numpy'):
            tx_pos = tx_pos.numpy()

        # RX positions for this TX, aligned with CFR user axis
        if per_tx_users_only:
            start_idx = tx_idx * num_users_per_tx
            end_idx = start_idx + num_users_per_tx
            rx_names = [f"UE_{i}" for i in range(start_idx, end_idx)]
        else:
            rx_names = [f"UE_{i}" for i in range(total_users)]

        import numpy as _np  # local import to avoid circulars

        rx_positions = []
        for rx_name in rx_names:
            rx = scene.get(rx_name)
            pos = rx.position
            if hasattr(pos, 'numpy'):
                pos = pos.numpy()
            else:
                pos = _np.array(pos)
            rx_positions.append(pos)
        rx_positions = _np.stack(rx_positions, axis=0)

        # Prepare per-TX metadata
        tx_metadata = {
            'tx_idx': tx_idx,
            'tx_name': tx_name,
            'tx_position': tx_pos,
            'rx_positions': rx_positions,
            'rx_names': rx_names,
            'num_txs': num_txs_actual,
            'num_users_per_tx': num_users_per_tx,
            'total_users': total_users,
            'num_sectors': num_sectors,
            'cfr_shape': h_tx.shape,
            'cfr_dtype': str(h_tx.dtype),
            'config': config,
        }

        yield {
            'tx_idx': tx_idx,
            'tx_name': tx_name,
            'h_tx': h_tx,
            'tx_metadata': tx_metadata,
        }
