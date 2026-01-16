import json
import numpy as np


def load_building_positions(building_positions_path):
    """loads building positions from .json file at building_positions_path"""
    with open(building_positions_path, 'r') as f:
        return json.load(f)

def get_antenna_positions(building_positions, scene_center=[0.0, 0.0], antenna_height_offset=10.0, shift_factor=0.5, num_deployment_buildings=1):
    """
    Find the highest building(s) and calculate heuristic-based antenna positions on the rooftops
    to minimize self-shadowing by shifting towards scene center.
    
    Args:
        building_positions: Dictionary of building data
        scene_center: Scene center coordinates [x, y] (from ground/scene bounds, not buildings). Defaults to [0.0, 0.0]
        antenna_height_offset: Height offset for antenna above rooftop (meters)
        shift_factor: Factor controlling shift towards center
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
        
        # Extract building data
        centroid = building_data['centroid']
        min_bounds = building_data['min']
        max_bounds = building_data['max']
        dimensions = building_data['dimensions']
        
        # Direction vector from centroid to scene center (normalized)
        dx = scene_center[0] - centroid[0]
        dy = scene_center[1] - centroid[1]
        direction_length = (dx**2 + dy**2)**0.5
        if direction_length > 0:
            dx /= direction_length
            dy /= direction_length
        
        # Shift magnitude based on roof size
        roof_size = (dimensions[0] + dimensions[1]) / 2
        shift_magnitude = roof_size * shift_factor
        
        # Calculate antenna position (shifted towards center, clamped to roof bounds)
        antenna_x = max(min_bounds[0], min(max_bounds[0], centroid[0] + dx * shift_magnitude))
        antenna_y = max(min_bounds[1], min(max_bounds[1], centroid[1] + dy * shift_magnitude))
        antenna_z = centroid[2] + antenna_height_offset
        
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
        print(f"Warning: Could not extract vertices from mesh: {e}")
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
            'has_rooftop': bool,
            'has_wall': bool
        }
        Same structure as load_building_positions() returns.
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
        
        if stats:
            building_data[building_id] = {
                'centroid': stats['centroid'],
                'min': stats['min'],
                'max': stats['max'],
                'dimensions': stats['dimensions'],
                'has_rooftop': 'rooftop' in parts,
                'has_wall': 'wall' in parts
            }
    
    return building_data