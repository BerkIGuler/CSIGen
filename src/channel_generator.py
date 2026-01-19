"""
Main API for channel generation from configuration.

This module provides the primary interface for generating channel data
using Sionna RT based on a configuration dictionary.
"""

from pathlib import Path
from typing import Dict

from src.scene_setup import setup_scene
from src.base_station import set_tx_antenna_array, add_base_station
from src.user_equipment import set_rx_antenna_array
from src.radio_map import solve_radio_map, sample_user_positions
from src.receivers import add_receivers_from_samples
from src.path_solver import solve_paths_per_tx
from src.channel import compute_cfr, save_channel_data


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
        - tx_power_dbm, bs_display_radius
        
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
        
        Optional:
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
    # Extract config with defaults
    scene_xml_path = Path(config['scene_xml_path'])
    carrier_frequency = config['carrier_frequency']
    
    # Scene setup parameters
    scene_center = config.get('scene_center', [0.0, 0.0])
    antenna_height_offset = config.get('antenna_height_offset', 10.0)
    num_deployment_buildings = config.get('num_deployment_buildings', 1)
    clip_terrain_to_buildings = config.get('clip_terrain_to_buildings', True)
    terrain_clip_margin = config.get('terrain_clip_margin', 15.0)
    user_shift_from_ground = config.get('user_shift_from_ground', 1.5)
    
    # Step 1: Setup scene
    print("Step 1: Setting up scene...")
    scene, building_positions, measurement_surface, antenna_information = setup_scene(
        scene_xml_path=scene_xml_path,
        carrier_frequency=carrier_frequency,
        scene_center=scene_center,
        antenna_height_offset=antenna_height_offset,
        num_deployment_buildings=num_deployment_buildings,
        clip_terrain_to_buildings=clip_terrain_to_buildings,
        terrain_clip_margin=terrain_clip_margin,
        user_shift_from_ground=user_shift_from_ground
    )
    
    # Step 2: Set antenna arrays
    print("\nStep 2: Setting antenna arrays...")
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
    print("\nStep 3: Adding base stations...")
    num_sectors = config['num_sectors']
    for i, (building_id, antenna_position) in enumerate(antenna_information):
        bs_name = f"BS_{i}"
        add_base_station(
            scene,
            bs_name,
            position=antenna_position,
            num_sectors=num_sectors,
            mechanical_tilt=config.get('mechanical_tilt', 10.0),
            azimuth_offset=config.get('azimuth_offset', 0.0),
            tx_power_dbm=config.get('tx_power_dbm', 43.0),
            display_radius=config.get('bs_display_radius', 15.0)
        )
    
    # Calculate total number of TXs
    num_txs = len(antenna_information) * num_sectors
    
    # Step 4: Solve radio map
    print("\nStep 4: Solving radio map...")
    radio_map = solve_radio_map(
        scene,
        measurement_surface=measurement_surface,
        diffuse_reflection=config.get('radio_map_diffuse_reflection', True),
        diffraction=config.get('radio_map_diffraction', True),
        edge_diffraction=config.get('radio_map_edge_diffraction', True),
        max_depth=config.get('radio_map_max_depth', 5),
        samples_per_tx=config.get('radio_map_samples_per_tx', 10**8)
    )
    
    # Step 5: Sample user positions
    print("\nStep 5: Sampling user positions...")
    sampled_positions = sample_user_positions(
        radio_map,
        num_pos_per_tx=config['num_user_samples_per_tx'],
        metric=config.get('user_sample_metric', 'path_gain'),
        min_val_db=config.get('user_sample_min_val_db', -150),
        tx_association=config.get('tx_association', True),
        center_pos=config.get('sample_center_pos', True),
        seed=config.get('user_sample_seed', 1)
    )
    
    # Step 6: Add receivers
    print("\nStep 6: Adding receivers...")
    num_txs_actual, num_users_per_tx, total_users = add_receivers_from_samples(
        scene,
        sampled_positions,
        num_sectors=num_sectors,
        mobility_preset=config['mobility_preset'],
        mobility_presets=config['mobility_presets'],
        seed=config.get('user_sample_seed', 1)
    )
    
    # Step 7: Solve paths per TX
    print("\nStep 7: Solving paths per TX...")
    paths_per_tx = solve_paths_per_tx(
        scene,
        num_txs=num_txs_actual,
        num_sectors=num_sectors,
        num_users_per_tx=num_users_per_tx,
        per_tx_users_only=config.get('path_solver_per_tx_users_only', True),
        max_depth=config.get('path_solver_max_depth', 5),
        max_num_paths_per_src=config.get('path_solver_max_num_paths_per_src', 10**6),
        samples_per_src=config.get('path_solver_samples_per_src', 10**6),
        synthetic_array=config.get('path_solver_synthetic_array', True),
        los=config.get('path_solver_los_mode', True),
        specular_reflection=config.get('path_solver_specular_reflection', True),
        diffuse_reflection=config.get('path_solver_diffuse_reflection', True),
        refraction=config.get('path_solver_refraction', True),
        diffraction=config.get('path_solver_diffraction', True),
        edge_diffraction=config.get('path_solver_edge_diffraction', True),
        diffraction_lit_region=config.get('path_solver_diffraction_lit_region', False),
        seed=config.get('path_solver_seed', 1)
    )
    
    # Step 8: Compute CFR
    print("\nStep 8: Computing Channel Frequency Response...")
    h = compute_cfr(
        paths_per_tx,
        num_subcarriers=config['num_subcarriers'],
        num_ofdm_symbols=config['num_ofdm_symbols'],
        subcarrier_spacing=config['subcarrier_spacing'],
        normalize_delays=config.get('cfr_normalize_delays', True),
        normalize=config.get('cfr_normalize', True),
        out_type=config.get('cfr_out_type', 'numpy')
    )
    
    # Prepare metadata
    metadata = {
        'config': config,
        'num_txs': num_txs_actual,
        'num_users_per_tx': num_users_per_tx,
        'total_users': total_users,
        'num_sectors': num_sectors,
        'h_shape': h.shape,
        'h_dtype': str(h.dtype)
    }
    
    print("\n✓ Channel generation complete!")
    
    return {
        'h': h,
        'metadata': metadata,
        'scene': scene,  # Optional: for inspection
        'radio_map': radio_map  # Optional: for inspection
    }
