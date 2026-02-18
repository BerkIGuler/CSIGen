import numpy as np
import colorsys
import logging
from typing import List
import matplotlib.pyplot as plt
import mitsuba as mi

logger = logging.getLogger(__name__)

def get_antenna_positions(
    building_positions, 
    scene_center=[0.0, 0.0],
    antenna_height_offset=10.0,
    num_deployment_buildings=1):
    """
    Find the highest building(s) and calculate antenna positions on the rooftops.
    The antenna is placed at the rooftop vertex closest to the scene center to minimize
    self-shadowing (the building mass is maximally behind the antenna).
    
    Args:
        building_positions: Dictionary of building data (must include 'rooftop_vertices')
        scene_center: Scene center coordinates [x, y]. Defaults to [0.0, 0.0]
        antenna_height_offset: Height offset for antenna above rooftop (meters)
        num_deployment_buildings: Number of buildings to deploy antennas on (default 1)
        
    Returns:
        list: List of tuples (building_id, antenna_position) where antenna_position is [x, y, z] (meters),
              ordered by building height (highest first). Returns empty list if no buildings.
    """
    if not building_positions:
        return []
    
    # Sort buildings by height (z coordinate of centroid) in descending order
    sorted_buildings = sorted(
        building_positions.items(),
        key=lambda item: item[1]['centroid'][2],
        reverse=True
    )
    
    # Limit to requested number of buildings (or available buildings)
    num_buildings = min(num_deployment_buildings, len(sorted_buildings))
    
    results = []
    for i in range(num_buildings):
        building_id, building_data = sorted_buildings[i]
        
        max_bounds = building_data['max']
        rooftop_vertices = building_data.get('rooftop_vertices')
        
        if rooftop_vertices is not None and len(rooftop_vertices) > 0:
            # Find the actual rooftop vertex closest to scene center
            distances = (rooftop_vertices[:, 0] - scene_center[0])**2 + \
                        (rooftop_vertices[:, 1] - scene_center[1])**2
            closest_idx = np.argmin(distances)
            antenna_x = float(rooftop_vertices[closest_idx, 0])
            antenna_y = float(rooftop_vertices[closest_idx, 1])
        else:
            # Fallback to centroid if no rooftop vertices available
            antenna_x = building_data['centroid'][0]
            antenna_y = building_data['centroid'][1]
            logger.warning(f"No rooftop vertices found for building {building_id}, using centroid instead")
        
        antenna_z = max_bounds[2] + antenna_height_offset
        
        results.append((building_id, [antenna_x, antenna_y, antenna_z]))
    
    return results


def get_scene_bounds(scene):
    """
    Get the bounding box of a scene.
    
    Args:
        scene: A Sionna scene object (assumes it has a Mitsuba scene)
        
    Returns:
        tuple: (bbox_min, bbox_max) where each is [x, y, z]
    """
    mi_scene = scene.mi_scene
    bbox = mi_scene.bbox()
    bbox_min = [bbox.min[0], bbox.min[1], bbox.min[2]]
    bbox_max = [bbox.max[0], bbox.max[1], bbox.max[2]]
    return bbox_min, bbox_max


def _extract_vertices_from_mi_mesh(mi_mesh):
    """
    Extract vertex positions from a Mitsuba mesh object.
    
    Args:
        mi_mesh: Mitsuba mesh object
        
    Returns:
        numpy.ndarray: Array of vertices with shape (N, 3) or None
    """
    try:
        # Get vertex positions buffer from Mitsuba mesh
        vertex_buffer = mi_mesh.vertex_positions_buffer()
        
        # Convert to numpy array
        # Mitsuba uses drjit, so we need to convert properly
        if hasattr(vertex_buffer, 'numpy'):
            vertices = np.array(vertex_buffer.numpy())
        else:
            # Try direct conversion
            vertices = np.array(vertex_buffer)
        
        # Reshape to (N, 3) if needed
        if len(vertices.shape) == 1:
            # Flatten array, reshape to (N, 3)
            vertices = vertices.reshape(-1, 3)
        
        return vertices
    except Exception as e:
        logger.warning(f"Warning: Could not extract vertices from mesh: {e}")
        return None


def _compute_stats_from_vertices(vertices):
    """
    Compute centroid, min, max, and dimensions from vertices.
    
    Args:
        vertices: numpy array of shape (N, 3) with vertex positions
        
    Returns:
        dict: Dictionary with 'centroid', 'min', 'max', 'dimensions' keys
    """
    if vertices is None or len(vertices) == 0:
        return None
    
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    z_coords = vertices[:, 2]
    
    centroid = [
        float(np.mean(x_coords)),
        float(np.mean(y_coords)),
        float(np.mean(z_coords))
    ]
    
    min_coords = [
        float(np.min(x_coords)),
        float(np.min(y_coords)),
        float(np.min(z_coords))
    ]
    
    max_coords = [
        float(np.max(x_coords)),
        float(np.max(y_coords)),
        float(np.max(z_coords))
    ]
    
    dimensions = [
        max_coords[0] - min_coords[0],
        max_coords[1] - min_coords[1],
        max_coords[2] - min_coords[2]
    ]
    
    return {
        'centroid': centroid,
        'min': min_coords,
        'max': max_coords,
        'dimensions': dimensions
    }


def _extract_rooftop_vertices(vertices, z_tolerance=0.5):
    """
    Extract vertices that are at the rooftop level (highest z values).
    
    Args:
        vertices: numpy array of shape (N, 3) with vertex positions
        z_tolerance: tolerance in meters for considering a vertex as "on the rooftop"
        
    Returns:
        numpy.ndarray: Array of rooftop vertices with shape (M, 2) containing only x, y coords,
                       or None if no vertices found
    """
    if vertices is None or len(vertices) == 0:
        return None
    
    max_z = np.max(vertices[:, 2])
    # Get vertices within tolerance of the max z (rooftop level)
    rooftop_mask = vertices[:, 2] >= (max_z - z_tolerance)
    rooftop_vertices = vertices[rooftop_mask]
    
    if len(rooftop_vertices) == 0:
        return None
    
    # Return only x, y coordinates (z is the rooftop height)
    return rooftop_vertices[:, :2]


def extract_building_positions_from_scene(scene):
    """
    Extract building positions and statistics directly from a loaded Sionna scene.
    
    Args:
        scene: A Sionna scene object (should be loaded with merge_shapes=False)
        
    Returns:
        dict: Dictionary mapping building_id -> {
            'centroid': [x, y, z],
            'min': [x, y, z],
            'max': [x, y, z],
            'dimensions': [width, depth, height],
            'rooftop_vertices': numpy array of shape (M, 2) with x, y coords of rooftop vertices,
            'has_rooftop': bool,
            'has_wall': bool
        }
    """
    building_data = {}
    
    # Group building objects by building ID
    buildings = {}
    
    for obj_name, obj in scene.objects.items():
        # Check if this is a building object
        # Sionna strips "mesh-" prefix, so objects are named "building_<id>_<part>"
        # Try patterns: "building_", or contains "building"
        building_match = None
        
        if obj_name.startswith('building_'):
            # Format: "building_<id>_<part>" (Sionna strips "mesh-" prefix)
            suffix = obj_name.replace('building_', '')
            building_match = suffix
        elif 'building' in obj_name.lower():
            # Try to extract building info from any name containing "building"
            # Look for pattern: ...building_<id>_<part>...
            import re
            match = re.search(r'building[_-](\d+)[_-](rooftop|wall)', obj_name.lower())
            if match:
                building_id = match.group(1)
                part_type = match.group(2)
                if building_id not in buildings:
                    buildings[building_id] = {}
                buildings[building_id][part_type] = obj
            continue
        
        if building_match is None:
            continue
        
        # Parse building ID and part type
        parts = building_match.split('_', 1)  # Split on first underscore
        
        if len(parts) < 2:
            continue
        
        building_id = parts[0]
        part_type = parts[1]  # 'rooftop' or 'wall'
        
        if building_id not in buildings:
            buildings[building_id] = {}
        
        buildings[building_id][part_type] = obj
    
    # Process each building
    for building_id, parts in sorted(buildings.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        # Prefer rooftop for stats, fallback to wall if no rooftop
        obj = parts.get('rooftop') or parts.get('wall')
        
        if obj is None:
            continue
        
        # Extract vertices from the mesh
        if not hasattr(obj, 'mi_mesh'):
            continue
        
        mi_mesh = obj.mi_mesh
        vertices = _extract_vertices_from_mi_mesh(mi_mesh)
        
        if vertices is None or len(vertices) == 0:
            continue
        
        # Compute statistics
        stats = _compute_stats_from_vertices(vertices)
        
        # Extract rooftop vertices for antenna placement
        rooftop_vertices = _extract_rooftop_vertices(vertices)
        
        if stats:
            building_data[building_id] = {
                'centroid': stats['centroid'],
                'min': stats['min'],
                'max': stats['max'],
                'dimensions': stats['dimensions'],
                'rooftop_vertices': rooftop_vertices,
                'has_rooftop': 'rooftop' in parts,
                'has_wall': 'wall' in parts
            }
    
    return building_data


def get_building_bounds(building_positions: dict) -> tuple:
    """
    Calculate the overall bounding box of all buildings.
    
    Args:
        building_positions: Dictionary from extract_building_positions_from_scene()
        
    Returns:
        tuple: ((min_x, min_y), (max_x, max_y)) or None if no buildings
    """
    if not building_positions:
        return None
    
    all_min_x = []
    all_min_y = []
    all_max_x = []
    all_max_y = []
    
    for building_id, data in building_positions.items():
        all_min_x.append(data['min'][0])
        all_min_y.append(data['min'][1])
        all_max_x.append(data['max'][0])
        all_max_y.append(data['max'][1])
    
    return (
        (min(all_min_x), min(all_min_y)),
        (max(all_max_x), max(all_max_y))
    )


def clip_terrain_to_buildings(scene, building_positions: dict, margin: float = 0.0):
    """
    Clip the terrain mesh vertices to the bounding box of buildings.
    
    This modifies the terrain mesh in-place by moving out-of-bounds vertices 
    to the nearest boundary (clamping approach).
    
    Args:
        scene: Sionna scene object
        building_positions: Dictionary from extract_building_positions_from_scene()
        margin: Extra margin in meters around the building bounds (default: 0)
        
    Returns:
        bool: True if clipping was performed, False otherwise
    """
    
    bounds = get_building_bounds(building_positions)
    if bounds is None:
        logger.warning("No buildings found, cannot clip terrain")
        return False
    
    (min_x, min_y), (max_x, max_y) = bounds
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    
    terrain_obj = scene.objects.get("ground")
    if terrain_obj is None:
        logger.warning("No ground object found in scene")
        return False
    
    if not hasattr(terrain_obj, 'mi_mesh'):
        logger.warning("Ground object is not a mesh")
        return False
    
    mi_mesh = terrain_obj.mi_mesh
    
    # Extract vertices
    vertices = _extract_vertices_from_mi_mesh(mi_mesh)
    if vertices is None:
        logger.warning("Could not extract vertices from terrain mesh")
        return False
    
    original_count = len(vertices)
    
    # Count vertices outside bounds
    outside_mask = (
        (vertices[:, 0] < min_x) | (vertices[:, 0] > max_x) |
        (vertices[:, 1] < min_y) | (vertices[:, 1] > max_y)
    )
    outside_count = np.sum(outside_mask)
    
    if outside_count == 0:
        logger.info("Terrain already within building bounds, no clipping needed")
        return True
    
    # Clamp vertices to bounds (move them to boundary instead of removing)
    vertices[:, 0] = np.clip(vertices[:, 0], min_x, max_x)
    vertices[:, 1] = np.clip(vertices[:, 1], min_y, max_y)
    
    # Update the mesh vertex positions
    try:
        params = mi.traverse(mi_mesh)
        params['vertex_positions'] = mi.Float(vertices.flatten())
        params.update()
        
        logger.info(f"Clipped terrain: clamped {outside_count}/{original_count} vertices to bounds")
        logger.info(f"  Bounds: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
        return True
        
    except Exception as e:
        logger.warning(f"Could not update mesh vertices: {e}")
        return False


def get_tx_color(tx_idx: int, num_txs: int) -> tuple:
    """
    Generate distinct color for each TX using HSV color space.
    Avoids red hues so TX markers remain visible on a red ground.

    Args:
        tx_idx: Index of the transmitter (0 to num_txs-1)
        num_txs: Total number of transmitters

    Returns:
        tuple: RGB color as (r, g, b) with values in [0, 1]
    """
    # Use hue in (0.08, 0.92) to skip red (hue 0 and 1)
    hue_min, hue_max = 0.08, 0.92
    denom = max(num_txs - 1, 1)
    hue = hue_min + (tx_idx / denom) * (hue_max - hue_min)
    saturation = 0.8
    value = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b)


def visualize_time_frequency_response(h_channel, title="Complex Channel Response"):
    """
    Create a figure visualizing a complex channel frequency response between a transmitter antenna element and a receiver antenna element.
    
    Creates two side-by-side plots showing magnitude and phase of the channel.
    Subcarriers are on the y-axis and OFDM symbols on the x-axis.
    
    Parameters
    ----------
    h_channel : np.ndarray
        Complex channel array with shape [num_ofdm_symbols, num_subcarriers]
    title : str
        Title for the plot
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plots. User can call fig.savefig() to save or plt.show() to display.
        
    Examples
    --------
    >>> h = paths.cfr(...)  # shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
    >>> sample_channel = h[0, 0, 0, 0, :, :]
    >>> fig = visualize_time_frequency_response(sample_channel, title="Channel Frequency Response")
    >>> fig.savefig("channel_response.png")  # Save to file
    >>> # or plt.show() to display interactively
    """
    h_channel = np.asarray(h_channel)
    
    if h_channel.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {h_channel.shape}")
    
    # Transpose to ensure: subcarriers on y-axis, OFDM symbols on x-axis
    # imshow treats first dim as rows (y) and second dim as columns (x)
    h_channel = h_channel.T  # Now shape is [num_subcarriers, num_ofdm_symbols]
    num_subcarriers, num_ofdm_symbols = h_channel.shape
    
    # Compute magnitude and phase
    magnitude = np.abs(h_channel)
    phase = np.angle(h_channel)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot magnitude (subcarriers on y-axis, OFDM symbols on x-axis)
    im1 = ax1.imshow(magnitude, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('OFDM Symbol Index')
    ax1.set_ylabel('Subcarrier Index')
    ax1.set_title(f'{title} - Magnitude')
    plt.colorbar(im1, ax=ax1, label='|H|')
    
    # Plot phase (subcarriers on y-axis, OFDM symbols on x-axis)
    im2 = ax2.imshow(phase, aspect='auto', origin='lower', cmap='hsv', interpolation='nearest')
    ax2.set_xlabel('OFDM Symbol Index')
    ax2.set_ylabel('Subcarrier Index')
    ax2.set_title(f'{title} - Phase')
    plt.colorbar(im2, ax=ax2, label='∠H (radians)')
    
    plt.tight_layout()
    
    return fig


def visualize_antenna_frequency_response(h_channel, title="Antenna Frequency Response"):
    """
    Create a figure visualizing a complex channel frequency response across antennas and subcarriers.
    
    Creates two side-by-side plots showing magnitude and phase of the channel.
    Subcarriers are on the y-axis and antenna indices on the x-axis.
    
    Parameters
    ----------
    h_channel : np.ndarray
        Complex channel array with shape [num_subcarriers, num_antennas] (or will be transposed to this).
        Assumes input already has subcarriers and antenna dimensions in correct order.
    title : str
        Title for the plot
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plots. User can call fig.savefig() to save or plt.show() to display.
        
    Examples
    --------
    >>> h = paths.cfr(...)  # shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
    >>> # Extract across antennas for one subcarrier: h[0, :, 0, 0, 0, :] -> [num_rx_ant, num_subcarriers]
    >>> # Or extract across subcarriers for one antenna: h[0, 0, 0, 0, 0, :] -> [num_subcarriers]
    >>> sample_channel = h[0, :, 0, 0, 0, :].T  # [num_subcarriers, num_rx_ant]
    >>> fig = visualize_antenna_frequency_response(sample_channel, title="Antenna Frequency Response")
    >>> fig.savefig("antenna_frequency_response.png")  # Save to file
    >>> # or plt.show() to display interactively
    """
    h_channel = np.asarray(h_channel)
    
    if h_channel.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {h_channel.shape}")
    
    # Compute magnitude and phase
    magnitude = np.abs(h_channel)
    phase = np.angle(h_channel)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot magnitude (subcarriers on y-axis, antennas on x-axis)
    im1 = ax1.imshow(magnitude, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('Subcarrier Index')
    ax1.set_ylabel('Antenna Index')
    ax1.set_title(f'{title} - Magnitude')
    plt.colorbar(im1, ax=ax1, label='|H|')
    
    # Plot phase (subcarriers on y-axis, antennas on x-axis)
    im2 = ax2.imshow(phase, aspect='auto', origin='lower', cmap='hsv', interpolation='nearest')
    ax2.set_xlabel('Subcarrier Index')
    ax2.set_ylabel('Antenna Index')
    ax2.set_title(f'{title} - Phase')
    plt.colorbar(im2, ax=ax2, label='∠H (radians)')
    
    plt.tight_layout()
    
    return fig


def display_path_count_histogram(hist: List[int]):
    """
    Display a single bar histogram: hist[i] = number of users with i valid paths (for one TX).

    Parameters
    ----------
    hist : list of int
        Histogram for one TX; hist[i] is the number of users with exactly i valid paths.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(hist)))
    ax.bar(x, hist, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Number of valid paths")
    ax.set_ylabel("Number of users")
    
    plt.tight_layout()
    plt.show()
