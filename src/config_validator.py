"""
Configuration validation module for channel generation.

This module provides validation for configuration parameters before they are
passed to the generate_channels API. Uses Pydantic for type-safe validation
with clear error messages.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# Allowed categorical values for configuration parameters
TX_RX_PATTERN = Literal["tr38901", "dipole", "iso", "hw_dipole"]
TX_RX_POLARIZATION = Literal["cross", "V", "H", "VH"]
USER_SAMPLE_METRIC = Literal["path_gain", "rss", "sinr"]
CFR_OUT_TYPE = Literal['drjit', 'jax', 'numpy', 'tf', 'torch']
MOBILITY_PRESET = Literal[
    "stationary",
    "stationary_to_tx",
    "pedestrian",
    "pedestrian_to_tx",
    "vehicular",
    "slow_walking",
    "fast_walking",
]


class MobilityPresetModel(BaseModel):
    """Model for a single mobility preset."""
    orientation_mode: Literal["random", "to_tx"] = Field(
        ..., description="Orientation mode for user equipment (allowed: random, to_tx)"
    )
    speed_distribution: Optional[Literal["pedestrian", "vehicular", "uniform_continuous"]] = Field(
        None, description="Speed distribution type for user movement (allowed: pedestrian, vehicular, uniform_continuous)"
    )
    direction_mode: Optional[Literal["random"]] = Field(
        None, description="Direction mode for movement (allowed: random)"
    )
    speed_min: Optional[float] = Field(None, ge=0.0, description="Minimum speed in m/s")
    speed_max: Optional[float] = Field(None, ge=0.0, description="Maximum speed in m/s")
    
    @model_validator(mode='after')
    def validate_speed_params(self):
        """Validate speed parameters based on speed_distribution."""
        # If one is provided but not the other, that's an error
        if (self.speed_min is None) != (self.speed_max is None):
            raise ValueError(
                "speed_min and speed_max must both be provided or both be None. "
                f"Got speed_min={self.speed_min}, speed_max={self.speed_max}"
            )
        
        # If uniform_continuous, both must be provided
        if self.speed_distribution == "uniform_continuous" and self.speed_min is None:
            raise ValueError(
                "speed_min and speed_max must both be provided when "
                f"speed_distribution is 'uniform_continuous'"
            )
        
        # If both are provided, validate the relationship
        if self.speed_min is not None and self.speed_max is not None:
            if self.speed_min >= self.speed_max:
                raise ValueError(
                    f"speed_min ({self.speed_min}) must be less than speed_max ({self.speed_max})"
                )
        
        return self


class ChannelConfigModel(BaseModel):
    """Pydantic model for validating channel generation configuration."""
    
    # Scene configuration
    scene_xml_path: Union[str, Path] = Field(..., description="Path to scene XML file")
    carrier_frequency: Union[int, float] = Field(..., gt=0, description="Carrier frequency in Hz")
    
    # Scene setup parameters
    scene_center: List[Union[int, float]] = Field(..., min_length=2, max_length=2, description="Center of the scene")
    antenna_height_offset: Union[int, float] = Field(..., ge=0, description="Height offset of antennas from the roof in meters")
    num_deployment_buildings: int = Field(..., ge=1, description="Number of buildings to deploy base stations on")
    clip_terrain_to_buildings: bool = Field(..., description="Whether to clip the terrain to building bounds")
    terrain_clip_margin: Union[int, float] = Field(..., ge=0, description="Margin in meters around buildings when clipping terrain")
    user_shift_from_ground: Union[int, float] = Field(..., ge=0, description="Up shift in meters of users from the ground plane")
    
    # TX antenna array parameters
    tx_num_rows: int = Field(..., ge=1, description="Number of rows in the tx antenna array")
    tx_num_cols: int = Field(..., ge=1, description="Number of columns in the tx antenna array")
    tx_vertical_spacing: float = Field(..., gt=0, description="Vertical spacing (in units of carrier wavelength)")
    tx_horizontal_spacing: float = Field(..., gt=0, description="Horizontal spacing (in units of carrier wavelength)")
    tx_pattern: TX_RX_PATTERN = Field(..., description="TX antenna pattern")
    tx_polarization: TX_RX_POLARIZATION = Field(..., description="Polarization of the TX antennas")
    
    # RX antenna array parameters
    rx_num_rows: int = Field(..., ge=1, description="Number of rows in the rx antenna array")
    rx_num_cols: int = Field(..., ge=1, description="Number of columns in the rx antenna array")
    rx_vertical_spacing: float = Field(..., gt=0, description="Vertical spacing (in units of carrier wavelength)")
    rx_horizontal_spacing: float = Field(..., gt=0, description="Horizontal spacing (in units of carrier wavelength)")
    rx_pattern: TX_RX_PATTERN = Field(..., description="RX antenna pattern")
    rx_polarization: TX_RX_POLARIZATION = Field(..., description="Polarization of the RX antennas")
    
    # Base station parameters
    num_sectors: int = Field(..., ge=1, description="Number of sectors for each base station")
    mechanical_tilt: Union[int, float] = Field(..., description="Mechanical tilt of the antenna in degrees")
    azimuth_offset: Union[int, float] = Field(..., description="Rotate the antenna arrays around z-axis by this amount in degrees")
    tx_power_dbm: Union[int, float] = Field(..., description="TX power in dBm for each TX antenna array")
    
    # Radio map solver parameters
    radio_map_specular_reflection: bool = Field(..., description="Whether to include specular reflection")
    radio_map_diffuse_reflection: bool = Field(..., description="Whether to include diffuse reflection")
    radio_map_refraction: bool = Field(..., description="Whether to include refraction")
    radio_map_diffraction: bool = Field(..., description="Whether to include diffraction")
    radio_map_edge_diffraction: bool = Field(..., description="Whether to include edge diffraction")
    radio_map_diffraction_lit_region: bool = Field(..., description="Whether to include diffraction in the lit region")
    radio_map_max_depth: int = Field(..., ge=1, description="Maximum number of ray scene interactions")
    radio_map_samples_per_tx: int = Field(..., ge=1, description="Number of samples per source")
    radio_map_seed: int = Field(..., description="Seed for radio map solver (reproducibility)")
    
    # User sampling parameters
    num_user_samples_per_tx: int = Field(..., ge=1, description="Number of user samples to generate per TX")
    user_sample_seed: int = Field(..., description="Seed for the user sampling")
    user_sample_min_val_db: Union[int, float] = Field(..., description="Minimum value in dB for the user sampling")
    user_sample_max_val_db: Union[int, float] = Field(..., description="Maximum value in dB for the user sampling")
    user_sample_min_dist: Union[int, float] = Field(..., description="Minimum distance in meters for the user sampling")
    user_sample_max_dist: Union[int, float] = Field(..., description="Maximum distance in meters for the user sampling")
    user_sample_metric: USER_SAMPLE_METRIC = Field(..., description="Metric for the user sampling")
    tx_association: bool = Field(
        ...,
        description=(
            "If True, only positions associated with a transmitter are chosen, "
            "i.e., positions where the chosen metric is the highest among all transmitters. "
            "Else, a user located in a sampled position for a specific transmitter may perceive "
            "a higher metric from another TX."
        ),
    )
    sample_center_pos: bool = Field(
        ...,
        description=(
            "If True, all returned positions are sampled from the cell center "
            "(i.e., the grid of the radio map). Otherwise, the positions are randomly drawn "
            "from the surface of the cell."
        ),
    )
    scene_edge_epsilon: Union[int, float] = Field(
        ...,
        ge=0.0,
        description=(
            "Minimum distance in meters from measurement surface edges to keep users. "
            "Users within this distance from any edge will be filtered out. "
            "Set to 0.0 to disable edge filtering."
        ),
    )
    
    # Mobility preset
    mobility_preset: MOBILITY_PRESET = Field(
        ..., description="Select preset to use (must be a key in mobility_presets)."
    )
    mobility_presets: Dict[str, MobilityPresetModel] = Field(..., description="Dictionary of mobility presets")
    
    # Path solver parameters
    path_solver_max_depth: int = Field(..., ge=1, description="Maximum number of ray scene interactions")
    path_solver_max_num_paths_per_src: int = Field(..., ge=1, description="Maximum number of paths per source")
    path_solver_samples_per_src: int = Field(..., ge=1, description="Number of samples per source")
    path_solver_synthetic_array: bool = Field(..., description="Use synthetic array for path computation")
    path_solver_los_mode: bool = Field(..., description="Enable line-of-sight paths")
    path_solver_specular_reflection: bool = Field(..., description="Include specular reflections")
    path_solver_diffuse_reflection: bool = Field(..., description="Include diffuse reflections")
    path_solver_refraction: bool = Field(..., description="Include refraction")
    path_solver_diffraction: bool = Field(..., description="Include diffraction")
    path_solver_edge_diffraction: bool = Field(..., description="Enables diffraction on free floating edges")
    path_solver_diffraction_lit_region: bool = Field(..., description="Enables diffraction in the lit region")
    path_solver_seed: int = Field(..., description="Seed for reproducibility")
    path_solver_per_tx_users_only: bool = Field(..., description="If true, solve paths only for users associated with each TX")
    
    # OFDM parameters
    num_subcarriers: int = Field(..., ge=1, description="Number of subcarriers")
    num_ofdm_symbols: int = Field(..., ge=1, description="Number of OFDM symbols")
    subcarrier_spacing: Union[int, float] = Field(..., gt=0, description="Spacing between subcarriers in Hz")
    
    # Channel Frequency Response (CFR) parameters
    cfr_normalize_delays: bool = Field(..., description="Whether to normalize delays in CFR computation")
    cfr_normalize: bool = Field(..., description="Whether to normalize the CFR")
    cfr_out_type: CFR_OUT_TYPE = Field(..., description="Output type for CFR")
    
    @field_validator('scene_xml_path')
    @classmethod
    def validate_scene_xml_path(cls, v):
        """Validate that scene XML path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Scene XML file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Scene XML path is not a file: {path}")
        return str(path)
    
    @field_validator('scene_center')
    @classmethod
    def validate_scene_center(cls, v):
        """Validate scene center is a list of 2 floats."""
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError(f"scene_center must be a list of 2 floats, got: {v}")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError(f"scene_center must contain only numbers, got: {v}")
        return [float(x) for x in v]
    
    @model_validator(mode='after')
    def validate_mobility_preset(self):
        """Validate that mobility_preset exists in mobility_presets."""
        if self.mobility_preset not in self.mobility_presets:
            available = list(self.mobility_presets.keys())
            raise ValueError(
                f"mobility_preset '{self.mobility_preset}' not found in mobility_presets. "
                f"Available presets: {available}"
            )
        return self
    
    class Config:
        """Pydantic configuration."""
        extra = 'ignore'
        validate_assignment = True


def validate_config(config: Dict) -> Dict:
    """
    Validate configuration dictionary before passing to generate_channels.
    
    This function validates all required parameters, their types, and constraints.
    It returns a validated dictionary that can be safely passed to generate_channels.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary to validate
        
    Returns
    -------
    dict
        Validated configuration dictionary
        
    Raises
    ------
    ValidationError
        If validation fails, with detailed error messages about what's wrong
    ValueError
        If there are logical errors in the configuration
    """
    try:
        # Convert nested mobility_presets dict to MobilityPresetModel instances
        if 'mobility_presets' in config:
            validated_presets = {}
            for key, preset_dict in config['mobility_presets'].items():
                validated_presets[key] = MobilityPresetModel(**preset_dict)
            config['mobility_presets'] = validated_presets
        
        # Create and validate the config model
        validated_model = ChannelConfigModel(**config)
        
        # Convert back to dict for compatibility with generate_channels
        validated_config = validated_model.model_dump()
        
        logger.info("Configuration validation successful!")
        return validated_config
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def load_validated_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from a YAML file, validate it, and return the validated config.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file
        
    Returns
    -------
    dict
        Validated configuration dictionary (ready to pass to generate_channels)
        
    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist
    ValidationError
        If validation fails
    """
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Configuration file is empty or invalid")
    
    return validate_config(config)
