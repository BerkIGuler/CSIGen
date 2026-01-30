"""
Sample script to run channel generation from a YAML config file.

This script demonstrates how to:
1. Load configuration from a YAML file
2. Generate channels using the generate_channels API
3. Save channel data and metadata using save_channel_data
"""

import argparse
import yaml
from pathlib import Path
import logging
import numpy as np

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
    
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading and validating configuration from {config_path}")
    validated_config = load_validated_config(config_path)
    
    logger.info("Starting channel generation...")
    result = generate_channels(validated_config)
    
    # Extract results
    cfr_per_tx = result['cfr_per_tx']
    metadata = result['metadata']
    scene = result['scene']

    num_txs = metadata['num_txs']
    num_users_per_tx = metadata['num_users_per_tx']
    total_users = metadata['total_users']
    num_sectors = metadata['num_sectors']
    per_tx_users_only = validated_config['path_solver_per_tx_users_only']
    
    # Save channel data for each TX
    logger.info(f"Saving channel data for {len(cfr_per_tx)} TX(s)...")
    for tx_idx, h_tx in enumerate(cfr_per_tx):
        output_path = output_dir / f"channel_tx_{tx_idx}.npz"

        # Derive TX name and position
        bs_id = tx_idx // num_sectors
        sector_id = (tx_idx % num_sectors) + 1
        tx_name = f"BS_{bs_id}_sector_{sector_id}"
        tx_obj = scene.get(tx_name)
        tx_pos = tx_obj.position
        if hasattr(tx_pos, 'numpy'):
            tx_pos = tx_pos.numpy()
        else:
            tx_pos = np.array(tx_pos)

        # RX positions for this TX, aligned with CFR user axis
        if per_tx_users_only:
            start_idx = tx_idx * num_users_per_tx
            end_idx = start_idx + num_users_per_tx
            rx_names = [f"UE_{i}" for i in range(start_idx, end_idx)]
        else:
            rx_names = [f"UE_{i}" for i in range(total_users)]

        rx_positions = []
        for rx_name in rx_names:
            rx = scene.get(rx_name)
            pos = rx.position
            if hasattr(pos, 'numpy'):
                pos = pos.numpy()
            else:
                pos = np.array(pos)
            rx_positions.append(pos)
        rx_positions = np.stack(rx_positions, axis=0)  # [num_users_per_tx or total_users, 3]

        # Create TX-specific metadata including positions
        tx_metadata = {
            'tx_idx': tx_idx,
            'tx_name': tx_name,
            'tx_position': tx_pos,
            'rx_positions': rx_positions,
            'rx_names': rx_names,
            'num_txs': num_txs,
            'num_users_per_tx': num_users_per_tx,
            'total_users': metadata['total_users'],
            'num_sectors': num_sectors,
            'cfr_shape': h_tx.shape,
            'cfr_dtype': str(h_tx.dtype),
            'config': metadata['config'],
        }

        # Save channel data + metadata (including positions)
        save_channel_data(
            h=h_tx,
            output_path=output_path,
            metadata=tx_metadata,
        )
        logger.info(f"Saved channel data for TX {tx_idx} ({tx_name}) to {output_path}")

    # Also save combined metadata summary
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml_metadata = {
            'num_txs': int(num_txs),
            'num_users_per_tx': int(num_users_per_tx),
            'total_users': int(metadata['total_users']),
            'num_sectors': int(num_sectors),
            'cfr_per_tx_shapes': [list(shape) for shape in metadata['cfr_per_tx_shapes']],
            'cfr_per_tx_dtypes': metadata['cfr_per_tx_dtypes'],
        }
        yaml.dump(yaml_metadata, f, default_flow_style=False)
    
    logger.info(f"Saved metadata summary to {metadata_path}")
    logger.info("Channel generation and saving complete!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
