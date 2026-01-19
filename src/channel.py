"""
Channel Frequency Response (CFR) computation and saving utilities.
"""

from sionna.rt import subcarrier_frequencies
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict


def compute_cfr(
    paths_per_tx: List,
    num_subcarriers: int,
    num_ofdm_symbols: int,
    subcarrier_spacing: float,
    normalize_delays: bool = True,
    normalize: bool = True,
    out_type: str = "numpy"
) -> np.ndarray:
    """
    Compute Channel Frequency Response (CFR) from paths.
    
    Parameters
    ----------
    paths_per_tx : list
        List of Paths objects (one per TX) from solve_paths_per_tx()
    num_subcarriers : int
        Number of subcarriers
    num_ofdm_symbols : int
        Number of OFDM symbols
    subcarrier_spacing : float
        Spacing between subcarriers in Hz
    normalize_delays : bool, default=True
        Whether to normalize delays in CFR computation
    normalize : bool, default=True
        Whether to normalize the CFR
    out_type : str, default="numpy"
        Output type ("numpy" or "tensorflow")
    
    Returns
    -------
    np.ndarray
        Combined CFR array with shape [total_users, num_txs, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]
        Note: When per_tx_users_only=True, this combines results from all TXs.
        The shape will be [total_users, num_txs, ...] where each TX's users are in their respective positions.
    """
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
    ofdm_symbol_duration = 1 / subcarrier_spacing
    
    # Compute CFR for each TX
    cfr_per_tx = []
    
    for tx_idx, paths_tx in enumerate(paths_per_tx):
        # Compute CFR for this TX
        h_tx = paths_tx.cfr(
            frequencies=frequencies,
            sampling_frequency=1/ofdm_symbol_duration,
            num_time_steps=num_ofdm_symbols,
            normalize_delays=normalize_delays,
            normalize=normalize,
            out_type=out_type
        )
        # h_tx shape: [num_selected_users, 1, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]
        cfr_per_tx.append(h_tx)
        print(f"TX {tx_idx}: CFR shape {h_tx.shape}")
    
    # Combine results from all TXs
    # Each h_tx has shape [num_selected_users, 1, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]
    # We need to combine them into [total_users, num_txs, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]
    
    # Get dimensions from first TX
    num_selected_users, _, num_rx_ant, num_tx_ant, num_subcarriers_check, num_ofdm_symbols_check = cfr_per_tx[0].shape
    num_txs = len(cfr_per_tx)
    total_users = num_selected_users * num_txs  # Assuming per_tx_users_only=True
    
    # Initialize combined array
    h = np.zeros(
        (total_users, num_txs, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols),
        dtype=cfr_per_tx[0].dtype
    )
    
    # Fill in the combined array
    for tx_idx, h_tx in enumerate(cfr_per_tx):
        start_user_idx = tx_idx * num_selected_users
        end_user_idx = (tx_idx + 1) * num_selected_users
        
        # Extract users for this TX: h_tx[start_user_idx:end_user_idx, 0, :, :, :, :]
        # But h_tx only has num_selected_users, so we slice it directly
        # h_tx shape: [num_selected_users, 1, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]
        # We want to place it at h[start_user_idx:end_user_idx, tx_idx, :, :, :, :]
        h[start_user_idx:end_user_idx, tx_idx, :, :, :, :] = h_tx[:, 0, :, :, :, :]
    
    # Print dimensions
    print(f"\nCFR shape from PathSolver: [num_users={total_users}, num_tx={num_txs}, num_rx_ant={num_rx_ant}, num_tx_ant={num_tx_ant}, num_subcarriers={num_subcarriers}, num_ofdm_symbols={num_ofdm_symbols}]")
    
    return h


def save_channel_data(
    h: np.ndarray,
    output_path: Path,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save channel data to file.
    
    Parameters
    ----------
    h : np.ndarray
        Channel frequency response array
    output_path : Path
        Path to save file (will save as .npz format)
    metadata : dict, optional
        Optional metadata dict to save alongside data
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare save dict
    save_dict = {
        'h': h,
        'shape': h.shape,
        'dtype': str(h.dtype)
    }
    
    # Add metadata if provided
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    # Save as .npz
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved channel data to {output_path}")
    print(f"  Shape: {h.shape}")
    print(f"  Dtype: {h.dtype}")
    if metadata:
        print(f"  Metadata keys: {list(metadata.keys())}")
