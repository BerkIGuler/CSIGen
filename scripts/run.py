"""
Sample script to run channel generation from a YAML config file.

This script demonstrates how to:
1. Load configuration from a YAML file
2. Generate channels using the generate_channels API
3. Save channel data and metadata using save_channel_data
"""

import argparse
import sys
from pathlib import Path

# Project root on path (script lives in scripts/, imports use top-level `src`)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
    
import yaml
from datetime import datetime
import logging

from src.channel_generator import generate_channels
from src.channel import save_channel_data
from src.config_validator import load_validated_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate channels from a YAML configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    return parser.parse_args()


def main():
    """Main function to run channel generation."""
    # Parse command-line arguments
    args = parse_args()
    
    config_path = Path(args.config)

    logger.info(f"Loading and validating configuration from {config_path}")
    validated_config = load_validated_config(config_path)

    # Derive scene name from the directory under 'scenes'
    scene_xml_path = Path(validated_config['scene_xml_path'])
    scene_name = scene_xml_path.parent.name or scene_xml_path.stem

    # Create a timestamped run directory: output/<scene_name>/<YYYYmmdd_HHMMSS>
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / scene_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting channel generation (streaming per TX)...")

    # Global summary information to be filled as we stream results
    num_txs = None
    num_users_per_tx = None
    total_users = None
    num_sectors = None
    cfr_shapes = []
    cfr_dtypes = []

    for item in generate_channels(validated_config):
        tx_idx = item['tx_idx']
        h_tx = item['h_tx']
        tx_metadata = item['tx_metadata']

        # Initialize global summary info from first TX
        if num_txs is None:
            num_txs = int(tx_metadata['num_txs'])
            num_users_per_tx = int(tx_metadata['num_users_per_tx'])
            total_users = int(tx_metadata['total_users'])
            num_sectors = int(tx_metadata['num_sectors'])

        cfr_shapes.append(list(h_tx.shape))
        cfr_dtypes.append(str(h_tx.dtype))

        output_path = output_dir / f"channel_tx_{tx_idx}.npz"
        # Save channel data + metadata (including positions)
        save_channel_data(
            h=h_tx,
            output_path=output_path,
            metadata=tx_metadata,
        )
        num_valid = tx_metadata.get('num_valid_channels', h_tx.shape[0])
        logger.info(
            "Saved channel data for TX %s (%s) to %s (%s valid channels)",
            tx_idx, tx_metadata['tx_name'], output_path, num_valid,
        )

    # Also save combined metadata summary (including the config and run info)
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml_metadata = {
            'scene_name': scene_name,
            'run_timestamp': timestamp,
            'num_txs': int(num_txs) if num_txs is not None else 0,
            'num_users_per_tx': int(num_users_per_tx) if num_users_per_tx is not None else 0,
            'total_users': int(total_users) if total_users is not None else 0,
            'num_sectors': int(num_sectors) if num_sectors is not None else 0,
            'cfr_per_tx_shapes': cfr_shapes,
            'cfr_per_tx_dtypes': cfr_dtypes,
            'config': validated_config,
        }
        yaml.dump(yaml_metadata, f, default_flow_style=False)
    
    logger.info(f"Saved metadata summary to {metadata_path}")
    logger.info("Channel generation and saving complete!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
