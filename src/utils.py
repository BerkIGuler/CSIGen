import json


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
        scene: A Sionna scene object
        
    Returns:
        tuple: (bbox_min, bbox_max) where each is [x, y, z], or (None, None) if unavailable
    """
    # Try scene bbox attribute first
    if hasattr(scene, 'bbox'):
        try:
            bbox = scene.bbox
            return list(bbox.min), list(bbox.max)
        except:
            pass
    
    # Fallback: access through Mitsuba scene
    if hasattr(scene, 'mi_scene'):
        try:
            mi_scene = scene.mi_scene
            bbox = mi_scene.bbox()
            bbox_min = [bbox.min[0], bbox.min[1], bbox.min[2]]
            bbox_max = [bbox.max[0], bbox.max[1], bbox.max[2]]
            return bbox_min, bbox_max
        except:
            pass
    
    return None, None