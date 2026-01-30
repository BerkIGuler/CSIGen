"""
Main API for channel generation from configuration.

This module provides the primary interface for generating channel data
using Sionna RT based on a configuration dictionary.
"""

import logging
from pathlib import Path
from typing import Dict

from src.scene_setup import setup_scene
from src.base_station import set_tx_antenna_array, add_base_station
from src.user_equipment import set_rx_antenna_array
from src.radio_map import solve_radio_map, sample_user_positions
from src.receivers import add_receivers_from_samples
from src.path_solver import solve_paths_per_tx
from src.channel import compute_cfr, save_channel_data

logger = logging.getLogger(__name__)

def generate_channels(config: Dict) -> Dict:
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
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'h': Channel frequency response array [total_users, num_txs, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]
        - 'metadata': Metadata dict containing config and scene info
        - 'scene': Scene object (optional, for inspection)
        - 'radio_map': RadioMap object (optional, for inspection)
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
    
    # Calculate total number of TXs
    num_txs = len(antenna_information) * num_sectors
    
    # Step 4: Solve radio map
    logger.info("Step 4: Solving radio map...")
    radio_map = solve_radio_map(
        scene,
        measurement_surface=measurement_surface,
        diffuse_reflection=config['radio_map_diffuse_reflection'],
        diffraction=config['radio_map_diffraction'],
        edge_diffraction=config['radio_map_edge_diffraction'],
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
        tx_association=config['tx_association'],
        center_pos=config['sample_center_pos'],
        seed=config['user_sample_seed']
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
    
    # Step 7: Solve paths per TX
    logger.info("Step 7: Solving paths per TX...")
    paths_per_tx = solve_paths_per_tx(
        scene,
        num_txs=num_txs_actual,
        num_sectors=num_sectors,
        num_users_per_tx=num_users_per_tx,
        per_tx_users_only=config['path_solver_per_tx_users_only'],
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
        seed=config['path_solver_seed']
    )
    
    # Step 8: Compute CFR
    logger.info("Step 8: Computing Channel Frequency Response...")
    cfr_per_tx = compute_cfr(
        paths_per_tx,
        num_subcarriers=config['num_subcarriers'],
        num_ofdm_symbols=config['num_ofdm_symbols'],
        subcarrier_spacing=config['subcarrier_spacing'],
        normalize_delays=config['cfr_normalize_delays'],
        normalize=config['cfr_normalize'],
        out_type=config['cfr_out_type']
    )
    
    # Prepare metadata
    metadata = {
        'config': config,
        'num_txs': num_txs_actual,
        'num_users_per_tx': num_users_per_tx,
        'total_users': total_users,
        'num_sectors': num_sectors,
        'cfr_per_tx_shapes': [h_tx.shape for h_tx in cfr_per_tx],
        'cfr_per_tx_dtypes': [str(h_tx.dtype) for h_tx in cfr_per_tx]
    }
    
    logger.info("Channel generation complete!")
    
    return {
        'cfr_per_tx': cfr_per_tx,
        'metadata': metadata,
        'scene': scene,  # Optional: for inspection
        'radio_map': radio_map  # Optional: for inspection
    }
