import numpy as np
from sionna.rt import PlanarArray, Receiver
from typing import List, Tuple


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
        
    Examples
    --------
    Set up standard UE array:
    >>> set_user_antenna_array(scene, num_rows=2, num_cols=2)
    
    Set up single antenna for IoT:
    >>> set_user_antenna_array(scene, num_rows=1, num_cols=1, pattern="iso")
    
    Set up advanced UE array:
    >>> set_user_antenna_array(scene, num_rows=2, num_cols=4, pattern="hw_dipole")
    """
    # Check if array already exists
    if hasattr(scene, 'rx_array') and scene.rx_array is not None and not force_overwrite:
        raise ValueError(
            "rx_array already exists on scene. Use force_overwrite=True to replace it, "
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
    scene.rx_array = antenna_array
    
    # Calculate total elements for reporting
    total_elements = num_rows * num_cols * 2 if polarization in ["cross", "VH"] else num_rows * num_cols
    
    print(f"UE Antenna Array Configuration Set:")
    print(f"  - Array: {num_rows}x{num_cols} ({total_elements} elements)")
    print(f"  - Pattern: {pattern}, Polarization: {polarization}")
    print(f"  - Spacing: V={vertical_spacing}λ, H={horizontal_spacing}λ")
    
    return antenna_array


def add_user(
    scene,
    name: str,
    position: List[float],
    # Mobility characteristics
    mobility_state: str = "pedestrian",  # "pedestrian", "vehicular"
    # Display options
    display_radius: float = 0.5,
    color: Tuple[float, float, float] = (0.0, 0.0, 1.0),  # Blue for users
) -> Receiver:
    """
    Add a user equipment (UE) receiver to a Sionna scene for downlink simulation.
    All devices are assumed to be handheld with characteristics determined by mobility state.
    
    Note: You must call set_user_antenna_array() before adding users to configure
    the antenna array that all users will share.
    
    Parameters
    ----------
    scene : sionna.rt.Scene
        The Sionna scene to add the user to
    name : str
        Name for the user receiver
    position : List[float]
        3D position [x, y, z] in meters
    mobility_state : str, default="pedestrian"
        Mobility classification determining velocity and orientation:
        - "pedestrian": random velocity 0.5-2 m/s, random orientation
        - "vehicular": random velocity 8-30 m/s, fixed orientation
    display_radius : float, default=0.5
        Display radius for visualization
    color : Tuple[float, float, float], default=(0.0, 0.0, 1.0)
        Color for visualization (RGB values 0.0-1.0)
    
    Returns
    -------
    Receiver
        Single receiver object for the UE
        
    Examples
    --------
    First set the UE antenna array configuration:
    >>> set_user_antenna_array(scene, num_rows=2, num_cols=2, pattern="iso")
    
    Then add users:
    >>> user1 = add_user(scene, "User_001", [50, 100, 1.5])
    >>> user2 = add_user(scene, "User_002", [200, 300, 1.6])
    
    Add high-mobility user (e.g., in vehicle):
    >>> user3 = add_user(scene, "User_003", [500, 400, 1.8], mobility_state="vehicular")
    """
    
    # Check that rx_array has been set
    if not hasattr(scene, 'rx_array') or scene.rx_array is None:
        raise ValueError(
            "No rx_array configured on scene. Call set_user_antenna_array() first to "
            "configure the antenna array that all users will share."
        )
    
    # Input validation
    if len(position) != 3:
        raise ValueError("Position must be [x, y, z]")
    
    # Calculate device orientation based on mobility state
    # Pedestrian -> random orientation, Vehicular -> fixed orientation
    orientation = _calculate_ue_orientation(mobility_state)
    
    # Calculate velocity based on mobility state
    velocity = _calculate_velocity(mobility_state)
    
    # Create receiver
    rx = Receiver(
        name=name,
        position=position,
        orientation=orientation,
        velocity=velocity,
        display_radius=display_radius,
        color=color
    )
    
    # Add to scene
    scene.add(rx)
    
    print(f"Added receiver '{name}' at position {position}")
    
    # Print brief summary
    velocity_magnitude = np.linalg.norm(velocity)
    print(f"  - Mobility: {mobility_state}, Velocity: {velocity_magnitude:.2f} m/s")
    
    return rx


def _calculate_ue_orientation(mobility_state: str) -> List[float]:
    """
    Calculate UE orientation based on mobility state.
    
    Parameters
    ----------
    mobility_state : str
        Either "pedestrian" or "vehicular"
    
    Returns
    -------
    List[float]
        Orientation [roll, pitch, azimuth] in radians
    """
    if mobility_state == "pedestrian":
        # Random orientation for handheld devices (pedestrians)
        azimuth = np.random.uniform(0, 2*np.pi)
        pitch = np.random.uniform(-np.pi/6, np.pi/6)  # ±30° tilt
        roll = np.random.uniform(-np.pi/4, np.pi/4)   # ±45° roll
        
    elif mobility_state == "vehicular":
        # Fixed orientation for vehicle-mounted devices
        azimuth = pitch = roll = 0.0
        
    else:
        # Default to pedestrian
        azimuth = np.random.uniform(0, 2*np.pi)
        pitch = np.random.uniform(-np.pi/6, np.pi/6)
        roll = np.random.uniform(-np.pi/4, np.pi/4)
    
    return [roll, pitch, azimuth]


def _calculate_velocity(mobility_state: str) -> List[float]:
    """
    Calculate random 3D velocity vector based on mobility state.
    
    Parameters
    ----------
    mobility_state : str
        Either "pedestrian" or "vehicular"
        - pedestrian: 0.5-2 m/s (1.8-7.2 km/h)
        - vehicular: 8-30 m/s (28.8-108 km/h)
    
    Returns
    -------
    List[float]
        3D velocity vector [vx, vy, vz] in m/s
    """
    if mobility_state == "pedestrian":
        # Pedestrian velocity: 0.5 - 2 m/s (1.8 - 7.2 km/h)
        speed = np.random.uniform(0.5, 2.0)
    elif mobility_state == "vehicular":
        # Vehicular velocity: 8 - 30 m/s (28.8 - 108 km/h)
        speed = np.random.uniform(8.0, 30.0)
    else:
        # Default to pedestrian
        speed = np.random.uniform(0.5, 2.0)
    
    # Random horizontal direction (azimuth)
    azimuth = np.random.uniform(0, 2*np.pi)
    
    # Calculate velocity components (assume motion in horizontal plane)
    vx = speed * np.cos(azimuth)
    vy = speed * np.sin(azimuth)
    vz = 0.0  # No vertical motion
    
    return [vx, vy, vz]


# Convenience functions for common mobility types

def add_pedestrian(scene, name: str, position: List[float], **kwargs) -> Receiver:
    """
    Add a pedestrian user with typical parameters.
    Uses pedestrian mobility: random orientation and velocity 0.5-2 m/s.
    """
    defaults = {
        'mobility_state': 'pedestrian', 
        'color': (0.0, 0.0, 1.0)  # Blue for pedestrian
    }
    defaults.update(kwargs)
    return add_user(scene, name, position, **defaults)


def add_vehicular(scene, name: str, position: List[float], **kwargs) -> Receiver:
    """
    Add a vehicular user with typical parameters.
    Uses vehicular mobility: fixed orientation and velocity 8-30 m/s.
    """
    defaults = {
        'mobility_state': 'vehicular',
        'color': (1.0, 1.0, 0.0)  # Yellow for vehicular
    }
    defaults.update(kwargs)
    return add_user(scene, name, position, **defaults)


# Utility function for adding multiple users
def add_multiple_users(scene, user_configs: List[dict]) -> List[Receiver]:
    """
    Add multiple users to scene with different configurations.
    
    Note: set_user_antenna_array() must be called before this function.
    
    Parameters
    ----------
    scene : sionna.rt.Scene
        Scene to add users to
    user_configs : List[dict]
        List of user configuration dictionaries
        Each dict should have: name, position, and optional parameters
        mobility_state: "pedestrian" or "vehicular"
    
    Returns
    -------
    List[Receiver]
        List of all created receivers
        
    Examples
    --------
    First set the antenna array:
    >>> set_user_antenna_array(scene, num_rows=2, num_cols=2)
    
    Then add multiple users:
    >>> users = add_multiple_users(scene, [
    ...     {"name": "User_001", "position": [100, 200, 1.5]},
    ...     {"name": "User_002", "position": [150, 250, 1.6]},
    ...     {"name": "User_003", "position": [300, 400, 1.8], "mobility_state": "vehicular"}
    ... ])
    """
    
    # Check that rx_array has been set
    if not hasattr(scene, 'rx_array') or scene.rx_array is None:
        raise ValueError(
            "No rx_array configured on scene. Call set_user_antenna_array() first."
        )
    
    receivers = []
    
    for config in user_configs:
        name = config.pop('name')
        position = config.pop('position')
        
        # Add user with remaining config as kwargs
        rx = add_user(scene, name, position, **config)
        receivers.append(rx)
    
    print(f"\nAdded {len(receivers)} users to scene with shared antenna array")
    return receivers
