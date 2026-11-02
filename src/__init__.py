"""
CSIGen - Channel generation package using Sionna RT.

The names below and the submodules are imported when they are first used. Thus,
the functions that read saved path data (for example ``load_cfr``) and the config
validation work without Sionna RT.
"""

import importlib as _importlib
import pkgutil as _pkgutil
from typing import TYPE_CHECKING

__version__ = "0.4.2"

# Public name -> module that defines it
_EXPORTS = {
    'generate_channels': 'src.channel_generator',
    'compute_cfr': 'src.channel',
    'save_channel_data': 'src.channel',
    'setup_scene': 'src.scene_setup',
    'solve_radio_map': 'src.radio_map',
    'sample_user_positions': 'src.radio_map',
    'add_receivers_from_samples': 'src.receivers',
    'rx_names_for_tx': 'src.receivers',
    'iter_paths_per_tx': 'src.path_solver',
    'iter_paths_for_receivers': 'src.path_solver',
    'validate_config': 'src.config_validator',
    'load_validated_config': 'src.config_validator',
    'cfr_from_paths': 'src.path_data',
    'load_cfr': 'src.path_data',
    'path_stats': 'src.path_data',
    'retarget_arrays': 'src.path_data',
    'retarget_velocities': 'src.path_data',
    'select_paths': 'src.path_data',
}
_SUBMODULES = {m.name for m in _pkgutil.iter_modules(__path__)}

__all__ = ['__version__', *_EXPORTS]

if TYPE_CHECKING:
    from src.channel_generator import generate_channels
    from src.channel import compute_cfr, save_channel_data
    from src.scene_setup import setup_scene
    from src.radio_map import solve_radio_map, sample_user_positions
    from src.receivers import add_receivers_from_samples, rx_names_for_tx
    from src.path_solver import iter_paths_per_tx, iter_paths_for_receivers
    from src.config_validator import validate_config, load_validated_config
    from src.path_data import (cfr_from_paths, load_cfr, path_stats, retarget_arrays, retarget_velocities,
                                select_paths)


def __getattr__(name):
    if name in _EXPORTS:
        module, attr = _EXPORTS[name], name
    elif name in _SUBMODULES:
        module, attr = f"{__name__}.{name}", None
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = _importlib.import_module(module)
    except ImportError as err:
        # An AttributeError keeps hasattr(), help() and IDE introspection working
        raise AttributeError(f"{__name__}.{name} needs a package that is not installed: {err}") from err
    if attr is not None:
        value = getattr(value, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(__all__) | _SUBMODULES)
