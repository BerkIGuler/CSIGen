import numpy as np
from sionna.rt import PlanarArray, Transmitter
from typing import List, Tuple, Union
import mitsuba as mi
import logging

logger = logging.getLogger(__name__)

def set_tx_antenna_array(
    scene,
    num_rows: int = 4,
    num_cols: int = 4,
    vertical_spacing: float = 0.5,
    horizontal_spacing: float = 0.5,
    pattern: str = "tr38901",
    polarization: str = "cross",
    force_overwrite: bool = False,
) -> None:
    """
    Set the transmit antenna array configuration for the scene.
    All base stations added to the scene will share this antenna array configuration.
    
    Parameters
    ----------
    scene : sionna.rt.Scene
        The Sionna scene to configure
    num_rows : int, default=8
        Number of antenna rows
    num_cols : int, default=8  
        Number of antenna columns
    vertical_spacing : float, default=0.5
        Vertical antenna spacing in wavelengths
    horizontal_spacing : float, default=0.5
        Horizontal antenna spacing in wavelengths
    pattern : str, default="tr38901"
        Antenna element pattern ("tr38901", "dipole", "iso", "hw_dipole")
    polarization : str, default="cross"
        Polarization type ("cross", "V", "H", "VH")
    force_overwrite : bool, default=False
        If True, overwrites existing tx_array without warning
        
    Returns
    -------
    PlanarArray
        The created antenna array object
        
    Examples
    --------
    Set up standard 64T64R array:
    >>> set_antenna_array(scene, num_rows=8, num_cols=8)
    
    Set up mmWave array:
    >>> set_antenna_array(scene, num_rows=16, num_cols=16)
    """
    # Check if array already exists
    if hasattr(scene, 'tx_array') and scene.tx_array is not None and not force_overwrite:
        raise ValueError(
            "tx_array already exists on scene. Use force_overwrite=True to replace it, "
            "or skip this call if you want to use the existing array configuration."
        )
    
    # Create antenna array
    antenna_array = PlanarArray(
        num_rows=num_rows,
        num_cols=num_cols,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        pattern=pattern,
        polarization=polarization
    )
    
    # Set on scene
    scene.tx_array = antenna_array
    
    # Calculate total elements for reporting
    total_elements = num_rows * num_cols * 2 if polarization in ["cross", "VH"] else num_rows * num_cols
    
    # Estimate beamwidth
    frequency_ghz = scene.frequency / 1e9  # Convert Hz to GHz
    estimated_beamwidth = estimate_array_beamwidth(
        num_cols=num_cols,
        frequency_ghz=frequency_ghz
    )
    
    logger.info("Antenna Array Configuration Set:")
    logger.info(f"  - Array: {num_rows}x{num_cols} ({total_elements} elements)")
    logger.info(f"  - Pattern: {pattern}, Polarization: {polarization}")
    logger.info(f"  - Spacing: V={vertical_spacing}λ, H={horizontal_spacing}λ")
    logger.info(f"  - Estimated beamwidth: {estimated_beamwidth}°")
    

def add_base_station(
    scene,
    name: str,
    position: Union[List[float], Tuple[float, float, float], mi.Point3f],
    num_sectors: int = 3,
    mechanical_tilt: float = 0.0,
    azimuth_offset: float = 0.0,
    tx_power_dbm: float = 43.0,
    display_radius: float = 2.0,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> List[Transmitter]:
    """
    Add a multi-sector base station to a Sionna scene.
    
    Note: You must call set_antenna_array() before adding base stations to configure
    the antenna array that all base stations will share.
    
    Parameters
    ----------
    scene : sionna.rt.Scene
        The Sionna scene to add the base station to
    name : str
        Base name for the base station (sectors will be named as name_sector_1, etc.)
    position : list, tuple, or mi.Point3f
        3D position [x, y, z] in meters
    num_sectors : int, default=3
        Number of sectors (typically 1, 3, or 6)
    mechanical_tilt : float, default=0.0
        Mechanical downtilt in degrees (positive = tilted down)
    azimuth_offset : float, default=0.0
        Overall azimuth rotation offset in degrees
    tx_power_dbm : float, default=43.0
        Transmit power in dBm (typical 5G: 40-46 dBm).
        TX power is the actual electrical power delivered to the antenna
        by the radio transmitter, measured in dBm (decibels relative to 1 milliwatt)
    display_radius : float, default=2.0
        Display radius for visualization of the base station as a sphere
    color : Tuple[float, float, float], default=(1.0, 0.0, 0.0)
        Color for visualization (RGB values between 0.0 and 1.0)
    
    Returns
    -------
    List[Transmitter]
        List of transmitter objects (one per sector)
        
    Examples
    --------
    First set the antenna array configuration:
    >>> set_antenna_array(scene, num_rows=8, num_cols=8, pattern="tr38901")
    
    Then add base stations:
    >>> tx1 = add_base_station(scene, "BS_001", [100, 200, 30])
    >>> tx2 = add_base_station(scene, "BS_002", [500, 600, 30])
    
    Custom 6-sector configuration:
    >>> tx3 = add_base_station(
    ...     scene, "BS_003", [0, 0, 25], 
    ...     num_sectors=6, mechanical_tilt=10.0
    ... )
    """
    
    # Convert position to mi.Point3f if it's a list or tuple
    if isinstance(position, (list, tuple)):
        if len(position) != 3:
            raise ValueError(f"Position must have exactly 3 elements [x, y, z], got {len(position)}")
        position = mi.Point3f(position)
    elif not isinstance(position, mi.Point3f):
        raise TypeError(f"Position must be a list, tuple, or mi.Point3f, got {type(position)}")
    
    # Check that tx_array has been set
    if not hasattr(scene, 'tx_array') or scene.tx_array is None:
        raise ValueError(
            "No tx_array configured on scene. Call set_antenna_array() first to "
            "configure the antenna array that all base stations will share."
        )
    
    # Input validation
    if num_sectors < 1:
        raise ValueError("Number of sectors must be >= 1")
    
    # Auto-calculate sector angles
    if num_sectors == 1:
        sector_angles = [0]
    elif num_sectors == 3:
        sector_angles = [0, 120, 240]
    elif num_sectors == 6:
        sector_angles = [0, 60, 120, 180, 240, 300]
    else:
        # Evenly distribute sectors
        angle_step = 360 / num_sectors
        sector_angles = [i * angle_step for i in range(num_sectors)]
    
    # Create transmitters
    transmitters = []
    
    for i, sector_angle in enumerate(sector_angles):
        # Calculate final orientation from mechanical_tilt and azimuth_offset
        final_azimuth = sector_angle + azimuth_offset
        
        # Convert to radians and create mi.Point3f: [roll, pitch, yaw]
        sector_orientation = mi.Point3f(
            [float(np.radians(final_azimuth)),
            float(np.radians(-mechanical_tilt)),
            0]
        )
        
        # Create transmitter
        sector_name = f"{name}_sector_{i+1}"
        
        tx = Transmitter(
            name=sector_name,
            position=position,
            orientation=sector_orientation,
            display_radius=display_radius,
            color=color,
            power_dbm=tx_power_dbm
        )
        
        # Add to scene
        scene.add(tx)
        transmitters.append(tx)
        
        logger.info(f"Added {sector_name} at position {position}, azimuth {final_azimuth:.1f}°")
    
    # Log summary
    logger.info(f"\nBase Station '{name}' Summary:")
    logger.info(f"  - Sectors: {num_sectors}")
    logger.info(f"  - Mechanical tilt: {mechanical_tilt:.1f}°")
    logger.info(f"  - TX Power: {tx_power_dbm} dBm")
    logger.info(f"  - Position: [{position[0]}, {position[1]}, {position[2]}] m")
    
    return transmitters

def estimate_array_beamwidth(num_cols: int, 
                           frequency_ghz: float = 3.5) -> float:
    """
    Estimate horizontal 3dB beamwidth of planar array.
    
    This is a rough approximation based on the standard antenna array beamwidth
    formula for uniform linear arrays. The base formula θ ≈ 51 * λ / D comes from
    the relationship between aperture size and beamwidth, where:
    - 51° is an empirical constant for 3dB beamwidth (derived from ~0.886 * 180/π)
    - λ is wavelength
    - D is aperture diameter (approximated as num_cols * spacing)
    
    Note: This is a simplified approximation. Actual beamwidth depends on:
    - Element spacing and array geometry
    - Element radiation pattern
    - Beamforming weights
    - Frequency-dependent effects
    
    Parameters
    ----------
    num_cols : int
        Number of antenna columns
    frequency_ghz : float, default=3.5
        Frequency in GHz
    
    Returns
    -------
    float
        Estimated 3dB beamwidth in degrees (rough approximation)
    """
    wavelength = 0.3 / frequency_ghz  # meters (c = 300 m/μs)
    aperture_horizontal = num_cols * 0.5 * wavelength
    
    # Base beamwidth formula: θ ≈ 51 * λ / D (degrees)
    # This assumes uniform aperture illumination and is a rough approximation
    # The constant 51° comes from the relationship: beamwidth ≈ 0.886 * (180/π) * λ/D
    # for a uniform linear array's 3dB beamwidth
    beamwidth = 51 * wavelength / aperture_horizontal
    
    return beamwidth
