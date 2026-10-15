"""
CSIGen - Channel generation package using Sionna RT.
"""

__version__ = "0.3.6"

from src.channel_generator import generate_channels
from src.channel import compute_cfr, save_channel_data
from src.scene_setup import setup_scene
from src.radio_map import solve_radio_map, sample_user_positions
from src.receivers import add_receivers_from_samples, rx_names_for_tx
from src.path_solver import iter_paths_per_tx, iter_paths_for_receivers
from src.config_validator import validate_config, load_validated_config
from src.path_data import (cfr_from_paths, load_cfr, path_stats, retarget_arrays, retarget_velocities,
                            select_paths)

__all__ = [
    '__version__',
    'generate_channels',
    'compute_cfr',
    'save_channel_data',
    'setup_scene',
    'solve_radio_map',
    'sample_user_positions',
    'add_receivers_from_samples',
    'rx_names_for_tx',
    'iter_paths_per_tx',
    'iter_paths_for_receivers',
    'validate_config',
    'load_validated_config',
    'cfr_from_paths',
    'load_cfr',
    'path_stats',
    'retarget_arrays',
    'retarget_velocities',
    'select_paths',
]
