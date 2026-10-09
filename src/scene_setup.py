"""
Scene setup utilities for loading and preparing scenes for channel generation.
"""

from pathlib import Path
from sionna.rt import ITURadioMaterial, load_scene, transform_mesh
from typing import Tuple, Optional, Any
import logging

from src.utils import (
    extract_building_positions_from_scene,
    get_antenna_positions,
    clip_terrain_to_buildings,
    get_building_bounds
)

logger = logging.getLogger(__name__)

SUPPORTED_GROUND_MATERIALS = ("concrete", "asphalt_concrete")


def setup_scene(
    scene_xml_path: Path,
    carrier_frequency: float,
    scene_center: list = [0.0, 0.0],
    antenna_height_offset: float = 10.0,
    num_deployment_buildings: int = 1,
    clip_terrain: bool = True,
    terrain_clip_margin: float = 15.0,
    user_shift_from_ground: float = 1.5,
    override_ground_material: Optional[str] = None,
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
    clip_terrain : bool, default=True
        Whether to clip the terrain to building bounds
    terrain_clip_margin : float, default=15.0
        Margin in meters around buildings when clipping terrain
    user_shift_from_ground : float, default=1.5
        Up shift in meters of users from the ground plane
    override_ground_material : str or None, default=None
        Optional ITU material type for the ground: "concrete" or "asphalt_concrete".
        The material is created if the scene does not define it.
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
    scene = load_scene(scene_xml_path, merge_shapes=merge_shapes)

    if override_ground_material is not None:
        _override_ground_material(scene, override_ground_material)
    
    # Set carrier frequency
    scene.frequency = carrier_frequency
    
    building_positions = extract_building_positions_from_scene(scene)
    
    # Optionally clip terrain to building bounds BEFORE creating measurement surface
    if clip_terrain and building_positions:
        bounds = get_building_bounds(building_positions)
        if bounds:
            (min_x, min_y), (max_x, max_y) = bounds
            logger.info(f"Building bounds: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
            clip_terrain_to_buildings(scene, building_positions, margin=terrain_clip_margin)
    
    # Detect if elevation data is esent by checking for ground object
    terrain_obj = scene.objects.get("ground")
    has_elevation = terrain_obj is not None
    
    # Create measurement surface from (potentially clipped) terrain
    if has_elevation:
        measurement_surface = terrain_obj.clone(as_mesh=True)
        transform_mesh(measurement_surface, translation=[0, 0, user_shift_from_ground])
    else:
        logger.warning("No elevation data found, cannot create measurement surface")
        measurement_surface = None
    
    # Get antenna positions for deployment buildings
    antenna_information = get_antenna_positions(
        building_positions,
        scene_center=scene_center,
        antenna_height_offset=antenna_height_offset,
        num_deployment_buildings=num_deployment_buildings
    )
    
    return scene, building_positions, measurement_surface, antenna_information


def _override_ground_material(scene, itu_type: str) -> None:
    """
    Assign the ITU material ``itu_type`` to the scene's ground object.

    The material is created with the thickness of the current ground material
    if the scene does not define it. Wet-ground materials are then removed,
    because Sionna RT evaluates every registered material when the frequency
    is set, and the ITU ground models are defined for 1-10 GHz only.
    """
    if itu_type not in SUPPORTED_GROUND_MATERIALS:
        raise ValueError(
            f"Unsupported override_ground_material: {itu_type}. "
            f"Supported values: None, {', '.join(repr(m) for m in SUPPORTED_GROUND_MATERIALS)}."
        )
    ground_obj = scene.objects.get("ground")
    if ground_obj is None:
        logger.warning(
            "Requested ground material override to %s, but no 'ground' object exists in scene.",
            itu_type,
        )
        return

    candidates = [f"itu_{itu_type}", f"mat-itu_{itu_type}"]
    material_name = next((name for name in candidates if scene.get(name) is not None), None)
    if material_name is None:
        material_name = candidates[0]
        thickness = float(ground_obj.radio_material.thickness[0])
        scene.add(ITURadioMaterial(material_name, itu_type, thickness=thickness))
        logger.info("Created ground material %s (thickness %.3f m)", material_name, thickness)
    
    # Assign the registered material to the ground mesh.
    ground_obj.radio_material = material_name

    for wet_name in ("wet_ground", "itu_wet_ground", "mat-itu_wet_ground"):
        if scene.get(wet_name) is not None:
            scene.remove(wet_name)
    logger.info("Overrode ground material to %s", material_name)
