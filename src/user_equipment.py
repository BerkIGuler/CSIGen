"""
User Equipment (UE) utilities for Sionna RT channel generation.

This module provides config-driven generation of UE parameters (orientations, velocities)
for batch data generation pipelines.
"""

import numpy as np
from sionna.rt import PlanarArray
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def set_rx_antenna_array(
    scene,
    num_rows: int = 2,
    num_cols: int = 2,
    vertical_spacing: float = 0.5,
    horizontal_spacing: float = 0.5,
    pattern: str = "hw_dipole",
    polarization: str = "VH",
    force_overwrite: bool = False,
) -> PlanarArray:
    """
    Set the receive antenna array configuration for the scene.
    All users added to the scene will share this antenna array configuration.
    
    Parameters
    ----------
    scene : sionna.rt.Scene
        The Sionna scene to configure
    num_rows : int, default=2
        Number of antenna rows (2x2 is typical for UE)
    num_cols : int, default=2  
        Number of antenna columns (2x2 is typical for UE)
    vertical_spacing : float, default=0.5
        Vertical antenna spacing in wavelengths
    horizontal_spacing : float, default=0.5
        Horizontal antenna spacing in wavelengths
    pattern : str, default="hw_dipole"
        Antenna element pattern ("iso", "dipole", "tr38901", "hw_dipole")
    polarization : str, default="VH"
        Polarization type ("V", "H", "VH", "cross")
    force_overwrite : bool, default=False
        If True, overwrites existing rx_array without warning
        
    Returns
    -------
    PlanarArray
        The created antenna array object
    """
    if hasattr(scene, 'rx_array') and scene.rx_array is not None and not force_overwrite:
        raise ValueError(
            "rx_array already exists on scene. Use force_overwrite=True to replace it, "
            "or skip this call if you want to use the existing array configuration."
        )
    
    antenna_array = PlanarArray(
        num_rows=num_rows,
        num_cols=num_cols,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        pattern=pattern,
        polarization=polarization
    )
    
    scene.rx_array = antenna_array
    
    total_elements = num_rows * num_cols * 2 if polarization in ["cross", "VH"] else num_rows * num_cols
    
    logger.info("UE Antenna Array Configuration Set:")
    logger.info(f"  - Array: {num_rows}x{num_cols} ({total_elements} elements)")
    logger.info(f"  - Pattern: {pattern}, Polarization: {polarization}")
    logger.info(f"  - Spacing: V={vertical_spacing}λ, H={horizontal_spacing}λ")
    
    return antenna_array


def generate_random_orientations(
    num_ues: int,
    seed: int = None
) -> np.ndarray:
    """
    Generate random orientations for a batch of user equipments.
    
    Simulates handheld device orientations with:
    - Yaw (azimuth): uniform [0, 2π]
    - Pitch: ±30° (device tilted up/down)
    - Roll: ±45° (device rotated in hand)
    
    Parameters
    ----------
    num_ues : int
        Number of UEs to generate orientations for
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Array of orientations with shape (num_ues, 3) where each row is 
        [roll, pitch, yaw] in radians following Sionna's convention
        
    Examples
    --------
    >>> orientations = generate_random_orientations(100, seed=42)
    """
    if seed is not None:
        np.random.seed(seed)
    
    yaw = np.random.uniform(0, 2 * np.pi, num_ues)
    pitch = np.random.uniform(-np.pi / 6, np.pi / 6, num_ues)  # ±30°
    roll = np.random.uniform(-np.pi / 4, np.pi / 4, num_ues)   # ±45°
    
    return np.stack([roll, pitch, yaw], axis=1)


def generate_ue_velocities(
    num_ues: int,
    speed_distribution: str = None,
    direction_mode: str = "random",
    speed_constant: float = 1.0,
    speed_min: float = 0.5,
    speed_max: float = 2.0,
    speed_values: List[float] = None,
    fixed_direction: float = 0.0,
    seed: int = None
) -> np.ndarray:
    """
    Generate velocities for a batch of user equipments with configurable distributions.
    
    Parameters
    ----------
    num_ues : int
        Number of UEs to generate velocities for
    speed_distribution : str, optional
        Speed distribution type:
        - None: All UEs have zero velocity (stationary)
        - "constant": All UEs have the same speed (speed_constant)
        - "uniform_continuous": Speed uniform in [speed_min, speed_max]
        - "uniform_discrete": Speed uniformly sampled from speed_values list
        - "pedestrian": Predefined 0.5-2 m/s range
        - "vehicular": Predefined 8-30 m/s range
    direction_mode : str, default="random"
        Direction mode:
        - "random": Random direction for each UE (uniform [0, 2π])
        - "fixed": All UEs move in fixed_direction
    speed_constant : float, default=1.0
        Speed value when speed_distribution="constant"
    speed_min : float, default=0.5
        Minimum speed for uniform_continuous distribution
    speed_max : float, default=2.0
        Maximum speed for uniform_continuous distribution
    speed_values : List[float], optional
        List of speed values for uniform_discrete distribution
    fixed_direction : float, default=0.0
        Direction angle in radians when direction_mode="fixed"
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Array of velocities with shape (num_ues, 3) where each row is [vx, vy, vz]
        
    Examples
    --------
    Stationary UEs:
    >>> velocities = generate_ue_velocities(100, speed_distribution=None)
    
    Pedestrian speeds with random direction:
    >>> velocities = generate_ue_velocities(100, speed_distribution="pedestrian")
    
    Constant speed, fixed direction (all moving east):
    >>> velocities = generate_ue_velocities(
    ...     100, speed_distribution="constant", speed_constant=5.0,
    ...     direction_mode="fixed", fixed_direction=0.0
    ... )
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Stationary UEs
    if speed_distribution is None:
        return np.zeros((num_ues, 3))
    
    # Generate speeds
    if speed_distribution == "constant":
        speeds = np.full(num_ues, speed_constant)
    elif speed_distribution == "uniform_continuous":
        speeds = np.random.uniform(speed_min, speed_max, num_ues)
    elif speed_distribution == "uniform_discrete":
        if speed_values is None or len(speed_values) == 0:
            raise ValueError("speed_values required for uniform_discrete distribution")
        speeds = np.random.choice(speed_values, num_ues)
    elif speed_distribution == "pedestrian":
        speeds = np.random.uniform(0.5, 2.0, num_ues)
    elif speed_distribution == "vehicular":
        speeds = np.random.uniform(8.0, 30.0, num_ues)
    else:
        raise ValueError(
            f"Unknown speed_distribution '{speed_distribution}'. "
            f"Must be None, 'constant', 'uniform_continuous', 'uniform_discrete', "
            f"'pedestrian', or 'vehicular'"
        )
    
    # Generate directions
    if direction_mode == "random":
        directions = np.random.uniform(0, 2 * np.pi, num_ues)
    elif direction_mode == "fixed":
        directions = np.full(num_ues, fixed_direction)
    else:
        raise ValueError(
            f"Unknown direction_mode '{direction_mode}'. Must be 'random' or 'fixed'"
        )
    
    # Convert to velocity vectors (horizontal plane, vz=0)
    vx = speeds * np.cos(directions)
    vy = speeds * np.sin(directions)
    vz = np.zeros(num_ues)
    
    return np.stack([vx, vy, vz], axis=1)


def generate_ue_parameters(
    num_ues: int,
    orientation_mode: str = "random",
    speed_distribution: str = None,
    direction_mode: str = "random",
    speed_constant: float = 1.0,
    speed_min: float = 0.5,
    speed_max: float = 2.0,
    speed_values: List[float] = None,
    fixed_direction: float = 0.0,
    seed: int = None
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Generate orientations and velocities for a batch of UEs.
    
    Parameters
    ----------
    num_ues : int
        Number of UEs
    orientation_mode : str, default="random"
        Orientation mode:
        - "random": Returns random orientations array
        - "to_tx": Returns None (use Sionna's look_at parameter instead)
    speed_distribution : str, optional
        Speed distribution (see generate_ue_velocities)
    direction_mode : str, default="random"
        Direction mode: "random" or "fixed"
    speed_constant, speed_min, speed_max, speed_values, fixed_direction
        Speed distribution parameters (see generate_ue_velocities)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    Tuple[Optional[np.ndarray], np.ndarray]
        - orientations: (num_ues, 3) array of [roll, pitch, yaw], or None if "to_tx"
        - velocities: (num_ues, 3) array of [vx, vy, vz]
        
    Examples
    --------
    Random orientations, stationary:
    >>> orientations, velocities = generate_ue_parameters(100, seed=42)
    
    Random orientations, pedestrian speeds:
    >>> orientations, velocities = generate_ue_parameters(
    ...     100, orientation_mode="random", speed_distribution="pedestrian", seed=42
    ... )
    
    To-TX orientation (use look_at when creating Receivers):
    >>> orientations, velocities = generate_ue_parameters(
    ...     100, orientation_mode="to_tx", speed_distribution="pedestrian", seed=42
    ... )
    >>> # orientations is None - use look_at parameter instead:
    >>> rx = Receiver(name="UE", position=pos, look_at=scene.get("BS_0"), velocity=velocities[i])
    """
    logger.info(f"Generating UE parameters for {num_ues} UEs")
    logger.info(f"  - Orientation mode: {orientation_mode}")
    logger.info(f"  - Speed distribution: {speed_distribution if speed_distribution else 'stationary'}")
    logger.info(f"  - Direction mode: {direction_mode}")
    if seed is not None:
        logger.info(f"  - Seed: {seed}")
    
    # Generate orientations
    if orientation_mode == "random":
        orientations = generate_random_orientations(num_ues, seed=seed)
        logger.info(f"Generated random orientations for {num_ues} UEs")
    elif orientation_mode == "to_tx":
        # Delegate to Sionna's look_at - return None
        orientations = None
        logger.info("Orientation mode 'to_tx': orientations set to None (use look_at parameter)")
    else:
        raise ValueError(
            f"Unknown orientation_mode '{orientation_mode}'. Must be 'random' or 'to_tx'"
        )
    
    # Generate velocities (use seed+1 to avoid correlation? - not sure if this is needed)
    velocity_seed = seed + 1 if seed is not None else None
    velocities = generate_ue_velocities(
        num_ues=num_ues,
        speed_distribution=speed_distribution,
        direction_mode=direction_mode,
        speed_constant=speed_constant,
        speed_min=speed_min,
        speed_max=speed_max,
        speed_values=speed_values,
        fixed_direction=fixed_direction,
        seed=velocity_seed
    )
    
    # Log velocity statistics
    speeds = np.linalg.norm(velocities, axis=1)
    logger.info(f"Generated velocities: min={np.min(speeds):.2f} m/s, max={np.max(speeds):.2f} m/s, mean={np.mean(speeds):.2f} m/s")
    
    return orientations, velocities
