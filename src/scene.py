import xml.etree.ElementTree as ET
import json
import struct
from pathlib import Path
from collections import defaultdict


def parse_ply_file(ply_path):
    """Parse a PLY file and extract vertex positions."""
    try:
        with open(ply_path, 'rb') as f:
            line = f.readline().decode('ascii').strip()
            if line != 'ply':
                return None
            
            format_type = None
            vertex_count = 0
            in_header = True
            
            while in_header:
                line = f.readline().decode('ascii').strip()
                if line.startswith('format'):
                    format_type = line.split()[1]
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line == 'end_header':
                    in_header = False
                    break
            
            if vertex_count == 0:
                return None
            
            if format_type == 'binary_little_endian':
                vertices = []
                vertex_format = '<ddd'
                vertex_size = struct.calcsize(vertex_format)
                
                for _ in range(vertex_count):
                    vertex_data = f.read(vertex_size)
                    if len(vertex_data) < vertex_size:
                        break
                    x, y, z = struct.unpack(vertex_format, vertex_data)
                    vertices.append([x, y, z])
                
                return vertices if vertices else None
            else:
                vertices = []
                for _ in range(vertex_count):
                    line = f.readline().decode('ascii').strip()
                    coords = [float(x) for x in line.split()[:3]]
                    vertices.append(coords)
                
                return vertices if vertices else None
                
    except Exception as e:
        print(f"Warning: Error parsing {ply_path}: {e}")
        return None


def compute_stats(vertices):
    """Compute centroid, min, max, and dimensions from vertices."""
    if not vertices:
        return None
    
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]
    
    centroid = [
        sum(x_coords) / len(x_coords),
        sum(y_coords) / len(y_coords),
        sum(z_coords) / len(z_coords)
    ]
    
    min_coords = [min(x_coords), min(y_coords), min(z_coords)]
    max_coords = [max(x_coords), max(y_coords), max(z_coords)]
    dimensions = [max_coords[i] - min_coords[i] for i in range(3)]
    
    return {
        'centroid': centroid,
        'min': min_coords,
        'max': max_coords,
        'dimensions': dimensions
    }


def process_scene(scene_dir):
    """Extract building positions from a single scene directory."""
    scene_path = Path(scene_dir)
    xml_path = scene_path / "scene.xml"
    
    if not xml_path.exists():
        return None
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    building_shapes = [s for s in root.findall(".//shape") 
                      if s.get('id', '').startswith('mesh-building')]
    
    buildings = defaultdict(dict)
    
    for shape in building_shapes:
        shape_id = shape.get('id')
        filename_elem = shape.find(".//string[@name='filename']")
        
        if filename_elem is None:
            continue
            
        filename = filename_elem.get('value')
        parts = shape_id.replace('mesh-building_', '').split('_')
        building_num = parts[0]
        part_type = parts[1] if len(parts) > 1 else 'unknown'
        
        buildings[building_num][part_type] = scene_path / filename
    
    building_data = {}
    
    for building_id, parts in sorted(buildings.items(), key=lambda x: int(x[0])):
        ply_file = parts.get('rooftop') or parts.get('wall')
        
        if ply_file and ply_file.exists():
            vertices = parse_ply_file(ply_file)
            
            if vertices and len(vertices) > 0:
                stats = compute_stats(vertices)
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


def process_scenes_directory(parent_dir):
    """Process all scene directories under a parent directory."""
    parent_path = Path(parent_dir)
    
    if not parent_path.exists():
        print(f"Error: Directory {parent_dir} does not exist")
        return
    
    scene_dirs = [d for d in parent_path.iterdir() 
                  if d.is_dir() and (d / "scene.xml").exists()]
    
    if not scene_dirs:
        print(f"No scene directories found in {parent_dir}")
        return
    
    print(f"Found {len(scene_dirs)} scene(s) to process\n")
    
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        print(f"Processing: {scene_name}")
        
        building_data = process_scene(scene_dir)
        
        if building_data:
            output_file = scene_dir / "buildings.json"
            with open(output_file, 'w') as f:
                json.dump(building_data, f, indent=2)
            
            print(f"  ✓ Extracted {len(building_data)} buildings → {output_file.name}\n")
        else:
            print(f"  ✗ No buildings found\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        parent_dir = sys.argv[1]
    else:
        parent_dir = "../scenes"
    
    print(f"Building Position Extractor")
    print(f"=" * 50)
    print(f"Scanning: {parent_dir}\n")
    
    process_scenes_directory(parent_dir)