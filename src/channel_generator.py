"""
Main API for channel generation from configuration.

This module provides the primary interface for generating channel data
using Sionna RT based on a configuration dictionary.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, Any

import numpy as np

from src.scene_setup import setup_scene
from src.base_station import set_tx_antenna_array, add_base_station
from src.user_equipment import set_rx_antenna_array
from src.radio_map import solve_radio_map, sample_user_positions, filter_positions_by_edge_distance
from src.receivers import add_receivers_from_samples, rx_names_for_tx
from src.path_solver import (
    iter_paths_for_receivers,
    set_specular_chain_table_size,
    get_valid_rx_mask,
    get_rx_los_nlos_mask,
    _compute_rx_valid_and_los_masks,
)
from src.channel import compute_cfr_for_paths

logger = logging.getLogger(__name__)

# Axes of the saved CFR tensor h (Sionna's Paths.cfr order)
CFR_AXES = ['rx', 'rx_ant', 'tx', 'tx_ant', 'ofdm_symbol', 'subcarrier']

# Antenna index of Sionna's PlanarArray: r is the row (0 = top), c the column
# along the array's local y axis and p the polarization, so the vertical index
# runs fastest.
ANTENNA_INDEX = 'p * num_rows * num_cols + c * num_rows + r'


def array_layout(config: Dict[str, Any], prefix: str, num_ant: int) -> Dict[str, Any]:
    """Describe the antenna array ``prefix`` ('tx' or 'rx') and its antenna index order."""
    num_rows = int(config[f'{prefix}_num_rows'])
    num_cols = int(config[f'{prefix}_num_cols'])
    return {
        'num_rows': num_rows,
        'num_cols': num_cols,
        'num_polarizations': int(num_ant) // (num_rows * num_cols),
        'num_ant': int(num_ant),
        'vertical_spacing': float(config[f'{prefix}_vertical_spacing']),
        'horizontal_spacing': float(config[f'{prefix}_horizontal_spacing']),
        'pattern': config[f'{prefix}_pattern'],
        'polarization': config[f'{prefix}_polarization'],
        'antenna_index': ANTENNA_INDEX,
    }


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
        - radio_map_max_depth, radio_map_samples_per_tx, radio_map_seed
        
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
        - path_solver_per_tx_users_only, path_solver_rx_batch_size, path_solver_spec_table_size
        
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
        - 'h_tx': np.ndarray (CFR for this TX; only valid channels)
        - 'tx_metadata': dict with TX/RX positions, global config-derived info,
          and per-channel metadata such as a LOS/NLOS indicator array
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
    override_ground_material = config.get('override_ground_material')
    
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
        user_shift_from_ground=user_shift_from_ground,
        override_ground_material=override_ground_material,
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
        samples_per_tx=config['radio_map_samples_per_tx'],
        seed=config['radio_map_seed'],
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
    num_txs_actual, users_per_tx, total_users = add_receivers_from_samples(
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
    rx_batch_size = config.get('path_solver_rx_batch_size')
    set_specular_chain_table_size(config.get('path_solver_spec_table_size'))
    all_rx_names = [f"UE_{i}" for i in range(total_users)]
    # Serving (sampling) TX of each user UE_k, as numbered by add_receivers_from_samples
    rx_serving_tx_all = np.repeat(np.arange(num_txs_actual), users_per_tx)

    for tx_idx in range(num_txs_actual):
        bs_id = tx_idx // num_sectors
        sector_id = (tx_idx % num_sectors) + 1
        tx_name = f"BS_{bs_id}_sector_{sector_id}"

        # Receivers for this TX (its own users or all users), solved in batches
        if per_tx_users_only:
            tx_rx_names = rx_names_for_tx(users_per_tx, tx_idx)
        else:
            tx_rx_names = all_rx_names
        if rx_batch_size and len(tx_rx_names) > rx_batch_size and config['cfr_out_type'] != 'numpy':
            raise ValueError("path_solver_rx_batch_size requires cfr_out_type 'numpy'.")

        h_parts, rx_names, los_parts = [], [], []
        num_total = 0
        buffer_fill = 0.0
        for paths_tx, row_rx_names, batch_fill in iter_paths_for_receivers(
            scene,
            tx_name=tx_name,
            rx_names=tx_rx_names,
            rx_batch_size=rx_batch_size,
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
        ):
            h_batch = compute_cfr_for_paths(
                paths_tx=paths_tx,
                num_subcarriers=config['num_subcarriers'],
                num_ofdm_symbols=config['num_ofdm_symbols'],
                subcarrier_spacing=config['subcarrier_spacing'],
                normalize_delays=config['cfr_normalize_delays'],
                normalize=config['cfr_normalize'],
                out_type=config['cfr_out_type'],
            )
            # Per-RX validity and LOS/NLOS determination (single pass over tau & interactions)
            valid_mask, rx_state_mask = _compute_rx_valid_and_los_masks(paths_tx)
            del paths_tx

            # Keep only receivers with at least one valid path
            h_parts.append(h_batch[valid_mask])
            rx_names.extend(name for name, ok in zip(row_rx_names, valid_mask) if ok)
            los_parts.append((rx_state_mask[valid_mask] == 1).astype(np.int32))
            num_total += len(valid_mask)
            buffer_fill = max(buffer_fill, batch_fill)

        if not h_parts:
            # No receivers for this TX (e.g. a sector without valid radio-map cells)
            logger.info("TX %s: no receivers to solve", tx_idx)
            h_parts = [np.zeros((0, scene.rx_array.num_ant, 1, scene.tx_array.num_ant,
                                 config['num_ofdm_symbols'], config['num_subcarriers']), dtype=np.complex64)]
            los_parts = [np.zeros(0, dtype=np.int32)]
        h_tx = h_parts[0] if len(h_parts) == 1 else np.concatenate(h_parts, axis=0)
        # Binary LOS indicator aligned with the CFR user axis
        los_binary = np.concatenate(los_parts)
        num_valid = len(rx_names)
        num_los = int(np.sum(los_binary))
        num_nlos = int(num_valid - num_los)
        if num_valid < num_total:
            logger.info(
                (
                    "TX %s: filtering out %s users with no valid paths "
                    "(keeping %s of %s; LoS=%s, NLoS=%s)"
                ),
                tx_idx,
                num_total - num_valid,
                num_valid,
                num_total,
                num_los,
                num_nlos,
            )

        tx_obj = scene.get(tx_name)
        tx_pos = tx_obj.position
        if hasattr(tx_pos, 'numpy'):
            tx_pos = tx_pos.numpy()

        # RX positions aligned with the CFR user axis
        rx_positions = []
        for rx_name in rx_names:
            pos = scene.get(rx_name).position
            rx_positions.append(pos.numpy() if hasattr(pos, 'numpy') else np.array(pos))
        # Same layout as Sionna's positions, [num_rx, 3, 1]
        rx_positions = np.stack(rx_positions, axis=0) if rx_positions else np.zeros((0, 3, 1), dtype=np.float32)

        # Prepare per-TX metadata (rx_positions/rx_names and h_tx are restricted to valid channels)
        tx_metadata = {
            'tx_idx': tx_idx,
            'tx_name': tx_name,
            'tx_position': tx_pos,
            'rx_positions': rx_positions,
            'rx_names': rx_names,
            # Per-valid-channel LOS indicator: 1 = LoS, 0 = NLoS
            'los_binary': los_binary,
            'num_txs': num_txs_actual,
            'users_per_tx': [int(n) for n in users_per_tx],
            # Serving (sampling) TX of each user, aligned with the CFR user axis
            'rx_serving_tx': rx_serving_tx_all[[int(n.split('_')[1]) for n in rx_names]].astype(np.int32),
            'total_users': total_users,
            'num_sectors': num_sectors,
            'num_valid_channels': num_valid,
            # Largest fill of Sionna's per-source path buffer over the solves
            # of this TX, as a fraction of max_num_paths_per_src; at 1.0 or
            # above, paths were discarded
            'path_buffer_fill': float(buffer_fill),
            'cfr_shape': h_tx.shape,
            'cfr_dtype': str(h_tx.dtype),
            'cfr_axes': CFR_AXES,
            'tx_array': array_layout(config, 'tx', scene.tx_array.num_ant),
            'rx_array': array_layout(config, 'rx', scene.rx_array.num_ant),
            'config': config,
        }

        yield {
            'tx_idx': tx_idx,
            'tx_name': tx_name,
            'h_tx': h_tx,
            'tx_metadata': tx_metadata,
        }
