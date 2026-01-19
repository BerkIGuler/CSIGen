# Package Structure Analysis

## Current Workflow (from notebook)

1. **Scene Setup** (Cell 2)
   - Load scene
   - Extract building positions
   - Clip terrain (optional)
   - Create measurement surface
   - Get antenna positions

2. **Antenna & Base Station Setup** (Cell 4)
   - Set TX antenna array
   - Set RX antenna array
   - Add base stations

3. **Radio Map Solving** (Cell 5)
   - Create RadioMapSolver
   - Solve radio map

4. **User Sampling** (Cell 9)
   - Sample user positions from radio map

5. **Receiver Creation** (Cell 11)
   - Generate UE parameters (orientations, velocities)
   - Create and add receivers

6. **Path Solving** (Cell 13)
   - Solve paths per TX (with temporary removal/restoration)

7. **CFR Computation** (Cell 14)
   - Compute CFR for each TX
   - Combine results

8. **Visualization** (Cell 15)
   - Visualize channels

## Existing Functions in `src/`

### `src/utils.py`
- ✅ `get_antenna_positions()` - Get antenna positions from buildings
- ✅ `get_scene_bounds()` - Get scene bounding box
- ✅ `extract_building_positions_from_scene()` - Extract building data
- ✅ `get_building_bounds()` - Get building bounds
- ✅ `clip_terrain_to_buildings()` - Clip terrain mesh
- ✅ `get_tx_color()` - Generate TX colors for visualization
- ✅ `visualize_complex_channel()` - Visualize channel response

### `src/base_station.py`
- ✅ `set_tx_antenna_array()` - Configure TX antenna array
- ✅ `add_base_station()` - Add multi-sector base station
- ✅ `estimate_array_beamwidth()` - Estimate beamwidth

### `src/user_equipment.py`
- ✅ `set_rx_antenna_array()` - Configure RX antenna array
- ✅ `generate_random_orientations()` - Generate random UE orientations
- ✅ `generate_ue_velocities()` - Generate UE velocities
- ✅ `generate_ue_parameters()` - Generate UE parameters (orientations + velocities)

## Missing Functions (Should be Created)

### 1. `src/scene_setup.py` (NEW FILE)
```python
def setup_scene(
    scene_xml_path: Path,
    carrier_frequency: float,
    clip_terrain_to_buildings: bool = True,
    terrain_clip_margin: float = 15.0,
    user_shift_from_ground: float = 1.5,
    merge_shapes: bool = False
) -> tuple:
    """
    Load and prepare scene for channel generation.
    
    Returns:
        (scene, building_positions, measurement_surface, antenna_information)
    """
    # Wraps Cell 2 logic
```

### 2. `src/radio_map.py` (NEW FILE)
```python
def solve_radio_map(
    scene,
    measurement_surface,
    diffuse_reflection: bool = True,
    diffraction: bool = True,
    edge_diffraction: bool = True,
    max_depth: int = 5,
    samples_per_tx: int = 10**8
) -> RadioMap:
    """
    Solve radio map for the scene.
    
    Returns:
        RadioMap object
    """
    # Wraps Cell 5 logic

def sample_user_positions(
    radio_map,
    num_pos_per_tx: int,
    metric: str = "path_gain",
    min_val_db: float = -150,
    tx_association: bool = True,
    center_pos: bool = True,
    seed: int = 1
) -> tuple:
    """
    Sample user positions from radio map.
    
    Returns:
        (positions, cell_ids) - both as tensors
    """
    # Wraps Cell 9 logic
```

### 3. `src/receivers.py` (NEW FILE)
```python
def add_receivers_from_samples(
    scene,
    sampled_positions: tuple,
    mobility_preset: str,
    mobility_presets: dict,
    num_sectors: int,
    seed: int = 1
) -> tuple:
    """
    Add receivers to scene from sampled positions.
    
    Returns:
        (num_txs, num_users_per_tx, total_users)
    """
    # Wraps Cell 11 logic
```

### 4. `src/path_solver.py` (NEW FILE)
```python
def solve_paths_per_tx(
    scene,
    num_txs: int,
    num_sectors: int,
    num_users_per_tx: int,
    per_tx_users_only: bool = True,
    max_depth: int = 5,
    max_num_paths_per_src: int = 10**6,
    samples_per_src: int = 10**6,
    synthetic_array: bool = True,
    los: bool = True,
    specular_reflection: bool = True,
    diffuse_reflection: bool = True,
    refraction: bool = True,
    diffraction: bool = True,
    edge_diffraction: bool = True,
    diffraction_lit_region: bool = False,
    seed: int = 1
) -> list:
    """
    Solve paths for each TX separately (computationally efficient).
    
    Returns:
        List of Paths objects (one per TX)
    """
    # Wraps Cell 13 logic
```

### 5. `src/channel.py` (NEW FILE)
```python
def compute_cfr(
    paths_per_tx: list,
    num_subcarriers: int,
    num_ofdm_symbols: int,
    subcarrier_spacing: float,
    normalize_delays: bool = True,
    normalize: bool = True,
    out_type: str = "numpy"
) -> np.ndarray:
    """
    Compute Channel Frequency Response (CFR) from paths.
    
    Returns:
        Combined CFR array with shape [total_users, num_txs, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]
    """
    # Wraps Cell 14 logic

def save_channel_data(
    h: np.ndarray,
    output_path: Path,
    metadata: dict = None
) -> None:
    """
    Save channel data to file (e.g., .npz or .h5).
    
    Args:
        h: Channel frequency response array
        output_path: Path to save file
        metadata: Optional metadata dict to save alongside data
    """
    # NEW - not in notebook yet
```

### 6. `src/channel_generator.py` (NEW FILE - MAIN API)
```python
def generate_channels(config: dict) -> dict:
    """
    Main function to generate channels from config.
    
    This is the primary API for the software package.
    Takes a config dictionary and returns channel data + metadata.
    
    Args:
        config: Configuration dictionary with all parameters
        
    Returns:
        dict with keys:
            - 'h': Channel frequency response array
            - 'metadata': Metadata dict (scene info, config, etc.)
            - 'scene': Scene object (optional)
            - 'radio_map': RadioMap object (optional)
    """
    # Orchestrates all the above functions
    # This is what users will call
```

## Proposed Package Structure

```
src/
├── __init__.py              # Package exports
├── base_station.py          # ✅ Already exists
├── user_equipment.py        # ✅ Already exists
├── utils.py                 # ✅ Already exists
├── scene_setup.py           # 🆕 NEW - Scene loading/preparation
├── radio_map.py             # 🆕 NEW - Radio map solving & sampling
├── receivers.py             # 🆕 NEW - Receiver creation
├── path_solver.py           # 🆕 NEW - Path solving per TX
├── channel.py               # 🆕 NEW - CFR computation & saving
└── channel_generator.py     # 🆕 NEW - Main API function
```

## Usage Example (Future)

```python
from src.channel_generator import generate_channels

# Load config (could be from YAML, JSON, or dict)
config = {
    "scene_xml_path": "../scenes/boston_1/scene.xml",
    "carrier_frequency": 3.5e9,
    "num_user_samples_per_tx": 1000,
    "mobility_preset": "pedestrian",
    "num_subcarriers": 120,
    "num_ofdm_symbols": 14,
    # ... all other config params
}

# Generate channels
result = generate_channels(config)

# Save data
from src.channel import save_channel_data
save_channel_data(
    result['h'],
    output_path=Path("./output/channels.npz"),
    metadata=result['metadata']
)
```

## Next Steps

1. **Create missing functions** - Refactor notebook cells into reusable functions
2. **Create config loader** - Support YAML/JSON config files
3. **Add saving functionality** - Save channel data + metadata
4. **Create main API** - `generate_channels()` function
5. **Add tests** - Unit tests for each function
6. **Documentation** - API docs, usage examples
