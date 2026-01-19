"""
Scene setup utilities for loading and preparing scenes for channel generation.
"""

from pathlib import Path
from sionna.rt import load_scene, transform_mesh
from typing import Tuple, Optional, Any

from src.utils import (
    extract_building_positions_from_scene,
    get_antenna_positions,
    clip_terrain_to_buildings,
    get_building_bounds
)


def setup_scene(
    scene_xml_path: Path,
    carrier_frequency: float,
    scene_center: list = [0.0, 0.0],
    antenna_height_offset: float = 10.0,
    num_deployment_buildings: int = 1,
    clip_terrain_to_buildings: bool = True,
    terrain_clip_margin: float = 15.0,
    user_shift_from_ground: float = 1.5,
    merge_shapes: bool = False
) -> Tuple[Any, dict, Optional[Any], list]:
    """
    Load and prepare scene for channel generation.
    
    This function:
    1. Loads the scene from XML
    2. Sets carrier frequency
    3. Extracts building positions
    4. Optionally clips terrain to building bounds
    5. Creates measurement surface from terrain (if elevation data exists)
    6. Gets antenna positions for deployment buildings
    
    Parameters
    ----------
    scene_xml_path : Path
        Path to scene XML file
    carrier_frequency : float
        Carrier frequency in Hz
    scene_center : list, default=[0.0, 0.0]
        Scene center coordinates [x, y]
    antenna_height_offset : float, default=10.0
        Height offset of antennas from the roof (meters)
    num_deployment_buildings : int, default=1
        Number of buildings to deploy base stations on
    clip_terrain_to_buildings : bool, default=True
        Whether to clip the terrain to building bounds
    terrain_clip_margin : float, default=15.0
        Margin in meters around buildings when clipping terrain
    user_shift_from_ground : float, default=1.5
        Up shift in meters of users from the ground plane
    merge_shapes : bool, default=False
        Whether to merge building shapes (False recommended for building extraction)
    
    Returns
    -------
    tuple
        (scene, building_positions, measurement_surface, antenna_information)
        - scene: Loaded Sionna scene object
        - building_positions: Dictionary of building data
        - measurement_surface: Mesh object for measurement surface (None if no elevation)
        - antenna_information: List of (building_id, antenna_position) tuples
    """
    # Load scene
    scene = load_scene(scene_xml_path, merge_shapes=merge_shapes)
    
    # Set carrier frequency
    scene.frequency = carrier_frequency
    
    # Extract building positions
    building_positions = extract_building_positions_from_scene(scene)
    
    # Optionally clip terrain to building bounds BEFORE creating measurement surface
    if clip_terrain_to_buildings and building_positions:
        bounds = get_building_bounds(building_positions)
        if bounds:
            (min_x, min_y), (max_x, max_y) = bounds
            print(f"Building bounds: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
            clip_terrain_to_buildings(scene, building_positions, margin=terrain_clip_margin)
    
    # Detect if elevation data is present by checking for ground object
    terrain_obj = scene.objects.get("ground")
    has_elevation = terrain_obj is not None
    
    # Create measurement surface from (potentially clipped) terrain
    if has_elevation:
        measurement_surface = terrain_obj.clone(as_mesh=True)
        transform_mesh(measurement_surface, translation=[0, 0, user_shift_from_ground])
    else:
        measurement_surface = None
    
    # Get antenna positions for deployment buildings
    antenna_information = get_antenna_positions(
        building_positions,
        scene_center=scene_center,
        antenna_height_offset=antenna_height_offset,
        num_deployment_buildings=num_deployment_buildings
    )
    
    return scene, building_positions, measurement_surface, antenna_information
