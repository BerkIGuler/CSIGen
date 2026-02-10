"""
Channel Frequency Response (CFR) computation and saving utilities.
"""

from sionna.rt import subcarrier_frequencies
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def compute_cfr_for_paths(
    paths_tx,
    num_subcarriers: int,
    num_ofdm_symbols: int,
    subcarrier_spacing: float,
    normalize_delays: bool = True,
    normalize: bool = False,
    out_type: str = "numpy",
) -> np.ndarray:
    """
    Compute Channel Frequency Response (CFR) from paths for a single TX.

    Parameters
    ----------
    paths_tx :
        Paths object for a single TX, typically from solve_paths_per_tx() or the
        streaming per-TX solver.
    num_subcarriers : int
        Number of subcarriers
    num_ofdm_symbols : int
        Number of OFDM symbols
    subcarrier_spacing : float
        Spacing between subcarriers in Hz
    normalize_delays : bool, default=True
        Whether to normalize delays in CFR computation
    normalize : bool, default=False
        Whether to normalize the CFR
    out_type : str, default=\"numpy\"
        Output type (\"numpy\" or \"tensorflow\")

    Returns
    -------
    np.ndarray
        CFR array for this TX with shape
        [num_selected_users, 1, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]
    """
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
    ofdm_symbol_duration = 1 / subcarrier_spacing

    h_tx = paths_tx.cfr(
        frequencies=frequencies,
        sampling_frequency=1 / ofdm_symbol_duration,
        num_time_steps=num_ofdm_symbols,
        normalize_delays=normalize_delays,
        normalize=normalize,
        out_type=out_type,
    )
    logger.info("Single TX CFR shape %s", h_tx.shape)
    return h_tx


def compute_cfr(
    paths_per_tx: List,
    num_subcarriers: int,
    num_ofdm_symbols: int,
    subcarrier_spacing: float,
    normalize_delays: bool = True,
    normalize: bool = False,
    out_type: str = "numpy",
) -> List[np.ndarray]:
    """
    Compute Channel Frequency Response (CFR) from paths for each TX.

    This function is retained for convenience and tests. For memory-efficient
    large-scale runs, prefer calling compute_cfr_for_paths per TX instead of
    materializing all CFRs at once.
    """
    cfr_per_tx: List[np.ndarray] = []
    for tx_idx, paths_tx in enumerate(paths_per_tx):
        h_tx = compute_cfr_for_paths(
            paths_tx=paths_tx,
            num_subcarriers=num_subcarriers,
            num_ofdm_symbols=num_ofdm_symbols,
            subcarrier_spacing=subcarrier_spacing,
            normalize_delays=normalize_delays,
            normalize=normalize,
            out_type=out_type,
        )
        cfr_per_tx.append(h_tx)
        logger.info("TX %s: CFR shape %s", tx_idx, h_tx.shape)

    if cfr_per_tx:
        num_rxs, num_rx_ant, num_txs, num_tx_ant, num_ofdm_symbols_check, num_subcarriers_check = cfr_per_tx[0].shape
        logger.info(
            "\\nCFR shape from PathSolver: [num_users=%s, num_tx=%s, num_rx_ant=%s, num_tx_ant=%s, num_subcarriers=%s, num_ofdm_symbols=%s]",
            num_rxs,
            num_txs,
            num_rx_ant,
            num_tx_ant,
            num_subcarriers_check,
            num_ofdm_symbols_check,
        )

    return cfr_per_tx


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
