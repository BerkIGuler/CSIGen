"""
Per-path data of the solved channels, the CFR computed from it, and per-channel path statistics.

The paths of all channels of one TX are stored in a flat (CSR) layout: the
paths of channel i are rows ``paths_offsets[i]:paths_offsets[i + 1]`` of the
other ``paths_*`` arrays, sorted by power, strongest first. The CFR saved in
``h`` is computed from exactly these arrays by ``cfr_from_paths``, so a file
without ``h`` can be rebuilt with ``load_cfr``, also for another numerology,
another planar array (``retarget_arrays``), another RX velocity
(``retarget_velocities``) or a subset of the paths (``select_paths``).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Per-path arrays, with their units and axes (P is the total number of paths of the file)
PATH_FIELDS = {
    "paths_offsets": "int64 [num_channels + 1]; the paths of channel i are rows offsets[i]:offsets[i + 1]",
    "paths_a": "complex64 [P, num_rx_ant, num_tx_ant]; path coefficient per antenna pair (passband, unitless)",
    "paths_tau": "float32 [P]; absolute delay (s)",
    "paths_doppler": "float32 [P]; Doppler shift (Hz)",
    "paths_theta_t": "float32 [P]; zenith angle of departure (rad)",
    "paths_phi_t": "float32 [P]; azimuth angle of departure (rad)",
    "paths_theta_r": "float32 [P]; zenith angle of arrival (rad)",
    "paths_phi_r": "float32 [P]; azimuth angle of arrival (rad)",
    "paths_interactions": "uint8 [P, max_depth]; interaction type per bounce "
                          "(0 none, 1 specular, 2 diffuse, 4 refraction, 8 diffraction)",
}
PATH_KEYS = tuple(PATH_FIELDS)

# Per-channel statistics saved in the metadata under 'path_stats', computed
# from all saved paths with the power of a path taken as its mean |a|^2 over
# antenna pairs
PATH_STATS = {
    "num_paths": "int32; number of paths",
    "path_gain_db": "float32; total power gain of all paths (dB)",
    "distance_m": "float32; 3D distance between the TX and the RX (m)",
    "first_delay_s": "float32; delay of the earliest path (s)",
    "mean_delay_s": "float32; power-weighted mean delay (s)",
    "delay_spread_s": "float32; power-weighted RMS delay spread (s)",
    "k_factor_db": "float32; power of the LoS path over the power of all other paths (dB); "
                   "NaN without a LoS path, inf if the LoS path is the only path",
    "asd_rad": "float32; power-weighted RMS azimuth spread of departure, around the circular mean (rad)",
    "asa_rad": "float32; power-weighted RMS azimuth spread of arrival, around the circular mean (rad)",
    "zsd_rad": "float32; power-weighted RMS zenith spread of departure (rad)",
    "zsa_rad": "float32; power-weighted RMS zenith spread of arrival (rad)",
    "dominant_power_fraction": "float32; power of the strongest path over the total power",
    "dominant_tau_s": "float32; delay of the strongest path (s)",
    "dominant_theta_t": "float32; zenith angle of departure of the strongest path (rad)",
    "dominant_phi_t": "float32; azimuth angle of departure of the strongest path (rad)",
    "dominant_theta_r": "float32; zenith angle of arrival of the strongest path (rad)",
    "dominant_phi_r": "float32; azimuth angle of arrival of the strongest path (rad)",
    "dominant_doppler_hz": "float32; Doppler shift of the strongest path (Hz)",
    "max_abs_doppler_hz": "float32; largest absolute Doppler shift over the paths (Hz)",
    "dominant_num_bounces": "float32; number of interactions of the strongest path (0 for LoS)",
    "max_num_bounces": "float32; largest number of interactions over the paths",
    "power_frac_los": "float32; fraction of the power in the LoS path",
    "power_frac_specular": "float32; fraction of the power in paths with at least one specular reflection",
    "power_frac_diffuse": "float32; fraction of the power in paths with at least one diffuse reflection",
    "power_frac_refraction": "float32; fraction of the power in paths with at least one refraction",
    "power_frac_diffraction": "float32; fraction of the power in paths with at least one diffraction",
}

# Interaction types by name, for select_paths and the power fractions of
# path_stats; the codes of Sionna RT's InteractionType, written out so that
# loading saved data does not need Sionna (extract_path_data checks them)
INTERACTION_NONE = 0
INTERACTION_TYPES = {
    "specular": 1,
    "diffuse": 2,
    "refraction": 4,
    "diffraction": 8,
}

# Largest number of elements of the [channels, paths, symbols, subcarriers]
# phase array per step of cfr_from_paths, to bound memory (2^24 complex128
# values are 256 MB)
_CFR_MAX_ELEMENTS = 2 ** 24

# Speed of light (m/s), as in Sionna RT
_SPEED_OF_LIGHT = 299792458.0

# Planar-array settings that retarget_arrays can change
_ARRAY_KEYS = ("num_rows", "num_cols", "vertical_spacing", "horizontal_spacing")


def extract_path_data(paths, rows: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Collect the valid paths of the receivers ``rows`` of one solve (one TX).

    Parameters
    ----------
    paths : sionna.rt.Paths
        Paths of one TX, traced with a synthetic array.
    rows : np.ndarray
        Boolean mask over the receivers of ``paths``; only these are kept.

    Returns
    -------
    dict
        The ``paths_*`` arrays (see PATH_FIELDS) of the kept receivers, in
        receiver order, with each receiver's paths sorted by power.
    """
    if not paths.synthetic_array:
        raise ValueError("Saving paths requires path_solver_synthetic_array: true")
    from sionna.rt.constants import InteractionType
    sionna_codes = {name: int(getattr(InteractionType, name.upper())) for name in INTERACTION_TYPES}
    if sionna_codes != INTERACTION_TYPES or int(InteractionType.NONE) != INTERACTION_NONE:
        raise RuntimeError(f"Sionna RT's interaction codes {sionna_codes} differ from INTERACTION_TYPES")
    rows = np.asarray(rows, dtype=bool)
    a_re, a_im = paths.a
    # [num_rx, num_rx_ant, 1, num_tx_ant, P] -> [num_rx, num_rx_ant, num_tx_ant, P]
    a = (a_re.numpy() + 1j * a_im.numpy()).astype(np.complex64)[:, :, 0]
    # Synthetic array: one value per path, [num_rx, 1, P] -> [num_rx, P]
    tau = paths.tau.numpy()[:, 0]
    fields = {
        "paths_doppler": paths.doppler.numpy()[:, 0],
        "paths_theta_t": paths.theta_t.numpy()[:, 0],
        "paths_phi_t": paths.phi_t.numpy()[:, 0],
        "paths_theta_r": paths.theta_r.numpy()[:, 0],
        "paths_phi_r": paths.phi_r.numpy()[:, 0],
    }
    # [max_depth, num_rx, 1, P] -> [num_rx, P, max_depth]
    interactions = np.moveaxis(paths.interactions.numpy()[:, :, 0], 0, -1)
    # Same validity criterion as the per-receiver masks of path_solver
    valid = paths.valid.numpy()[:, 0] & (tau > 0)
    # In float64, as _path_power, so that path_stats and select_paths see the same order
    power = np.mean(np.abs(a.astype(np.complex128)) ** 2, axis=(1, 2))

    order = []
    counts = []
    for r in np.flatnonzero(rows):
        idx = np.flatnonzero(valid[r])
        idx = idx[np.argsort(-power[r, idx], kind="stable")]
        order.append((r, idx))
        counts.append(len(idx))
    num_rx_ant, num_tx_ant = a.shape[1], a.shape[2]
    max_depth = interactions.shape[-1]

    def gather(x, tail=()):
        # x is [num_rx, P, ...]; the paths of receiver r are x[r][idx]
        if not order:
            return np.zeros((0,) + tail, dtype=x.dtype)
        return np.concatenate([x[r][idx] for r, idx in order])

    out = {
        "paths_offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        "paths_a": (np.concatenate([np.moveaxis(a[r][:, :, idx], -1, 0) for r, idx in order])
                    if order else np.zeros((0, num_rx_ant, num_tx_ant), dtype=np.complex64)),
        "paths_tau": gather(tau).astype(np.float32),
        **{k: gather(v).astype(np.float32) for k, v in fields.items()},
        "paths_interactions": gather(interactions, (max_depth,)).astype(np.uint8),
    }
    return out


def concat_path_data(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate the path data of consecutive receiver batches of one TX."""
    offsets = [np.zeros(1, dtype=np.int64)]
    base = 0
    for p in parts:
        offsets.append(p["paths_offsets"][1:] + base)
        base += int(p["paths_offsets"][-1])
    out = {"paths_offsets": np.concatenate(offsets)}
    for key in PATH_KEYS[1:]:
        out[key] = np.concatenate([p[key] for p in parts])
    return out


def empty_path_data(num_rx_ant: int, num_tx_ant: int, max_depth: int) -> Dict[str, np.ndarray]:
    """Path data of a TX without channels."""
    out = {key: np.zeros(0, dtype=np.float32) for key in PATH_KEYS}
    out["paths_offsets"] = np.zeros(1, dtype=np.int64)
    out["paths_a"] = np.zeros((0, num_rx_ant, num_tx_ant), dtype=np.complex64)
    out["paths_interactions"] = np.zeros((0, max_depth), dtype=np.uint8)
    return out


def slice_path_data(path_data: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Path data of the channels ``rows`` (integer indices), in that order."""
    offsets = path_data["paths_offsets"]
    idx = [np.arange(offsets[r], offsets[r + 1]) for r in rows]
    idx = np.concatenate(idx).astype(np.int64) if idx else np.zeros(0, dtype=np.int64)
    counts = np.array([offsets[r + 1] - offsets[r] for r in rows], dtype=np.int64)
    out = {"paths_offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)}
    for key in PATH_KEYS[1:]:
        out[key] = path_data[key][idx]
    return out


def _path_power(path_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Power of each path, the mean |a|^2 over antenna pairs, [P]."""
    return np.mean(np.abs(path_data["paths_a"].astype(np.complex128)) ** 2, axis=(1, 2))


def select_paths(path_data: Dict[str, np.ndarray], max_paths: Optional[int] = None,
                 power_range_db: Optional[float] = None, max_bounces: Optional[int] = None,
                 exclude_interactions: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Path data with only some of the paths of each channel.

    exclude_interactions drops the paths with at least one interaction of the
    given types (names of INTERACTION_TYPES); max_bounces drops the paths with
    more interactions (0 keeps only the LoS path); power_range_db drops the
    paths more than this many dB below the strongest path of the channel,
    taken over all its paths; max_paths then keeps the strongest remaining
    paths. A channel can end up without paths; its CFR is then zero.
    """
    keep = np.ones(len(path_data["paths_tau"]), dtype=bool)
    inter = path_data["paths_interactions"]
    if exclude_interactions:
        unknown = set(exclude_interactions) - set(INTERACTION_TYPES)
        if unknown:
            raise ValueError(f"Unknown interaction types {sorted(unknown)}; use {list(INTERACTION_TYPES)}")
        for name in exclude_interactions:
            keep &= ~np.any(inter == INTERACTION_TYPES[name], axis=1)
    if max_bounces is not None:
        keep &= np.sum(inter != INTERACTION_NONE, axis=1) <= max_bounces
    offsets = path_data["paths_offsets"]
    counts = np.diff(offsets)
    rows = np.repeat(np.arange(len(counts)), counts)
    if power_range_db is not None and len(rows):
        power = _path_power(path_data)
        strongest = np.zeros(len(counts))
        np.maximum.at(strongest, rows, power)
        keep &= power >= strongest[rows] * 10 ** (-power_range_db / 10)
    if max_paths is not None:
        # Paths are sorted by power, so the rank among the kept paths of a channel keeps that order
        kept_before = np.cumsum(keep) - keep
        rank = kept_before - kept_before[np.repeat(offsets[:-1], counts)]
        keep &= rank < max_paths
    new_counts = np.bincount(rows[keep], minlength=len(counts))
    out = {"paths_offsets": np.concatenate([[0], np.cumsum(new_counts)]).astype(np.int64)}
    for key in PATH_KEYS[1:]:
        out[key] = path_data[key][keep]
    return out


def subcarrier_offsets(num_subcarriers: int, subcarrier_spacing: float) -> np.ndarray:
    """Baseband subcarrier frequencies (Hz), as Sionna's subcarrier_frequencies."""
    if num_subcarriers % 2 == 0:
        start, limit = -num_subcarriers / 2, num_subcarriers / 2
    else:
        start, limit = -(num_subcarriers - 1) / 2, (num_subcarriers - 1) / 2 + 1
    return np.arange(start, limit, dtype=np.float64) * float(subcarrier_spacing)


def cfr_from_paths(
    path_data: Dict[str, np.ndarray],
    carrier_frequency: float,
    num_subcarriers: int,
    num_ofdm_symbols: int,
    subcarrier_spacing: float,
    normalize_delays: bool = False,
    normalize: bool = False,
    symbol_duration: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the CFR of every channel from its paths, in float64.

    h(f, t) = sum_l a_l exp(-j 2 pi (f_c + f) tau_l) exp(j 2 pi nu_l t), with
    f the subcarrier offsets and t = n * symbol_duration, as in Sionna's
    Paths.cfr. symbol_duration defaults to 1 / subcarrier_spacing (no cyclic
    prefix), the setting of generate_channels. With normalize_delays, the earliest delay of each channel is
    subtracted first; with normalize, each channel is scaled to unit average
    energy over antennas, symbols and subcarriers.

    Returns
    -------
    np.ndarray
        complex64 [num_channels, num_rx_ant, 1, num_tx_ant, num_ofdm_symbols, num_subcarriers]
    """
    offsets = path_data["paths_offsets"]
    num_ch = len(offsets) - 1
    _, num_rx_ant, num_tx_ant = path_data["paths_a"].shape
    freqs = subcarrier_offsets(num_subcarriers, subcarrier_spacing)
    if symbol_duration is None:
        symbol_duration = 1 / float(subcarrier_spacing)
    times = np.arange(num_ofdm_symbols, dtype=np.float64) * float(symbol_duration)
    fc = float(carrier_frequency)
    h = np.zeros((num_ch, num_rx_ant, 1, num_tx_ant, num_ofdm_symbols, num_subcarriers), dtype=np.complex64)
    counts = np.diff(offsets)

    grid = num_ofdm_symbols * num_subcarriers
    start = 0
    while start < num_ch:
        # Add channels while the padded phase array stays within the bound
        stop, p_max = start + 1, int(counts[start])
        while stop < num_ch and (stop + 1 - start) * max(p_max, int(counts[stop])) * grid <= _CFR_MAX_ELEMENTS:
            p_max = max(p_max, int(counts[stop]))
            stop += 1
        n = stop - start
        h_chunk = np.zeros((n, num_rx_ant, num_tx_ant, num_ofdm_symbols, num_subcarriers), dtype=np.complex128)
        if p_max > 0:
            # Pad the paths of the chunk to [n, p_max]; padded paths have a = 0
            sel = np.arange(offsets[start], offsets[stop])
            row = np.repeat(np.arange(n), counts[start:stop])
            col = sel - np.repeat(offsets[start:stop], counts[start:stop])
            a = np.zeros((n, num_rx_ant, num_tx_ant, p_max), dtype=np.complex128)
            a[row, :, :, col] = path_data["paths_a"][sel]
            tau = np.zeros((n, p_max))
            tau[row, col] = path_data["paths_tau"][sel]
            nu = np.zeros((n, p_max))
            nu[row, col] = path_data["paths_doppler"][sel]
            if normalize_delays:
                first = np.full(n, np.inf)
                np.minimum.at(first, row, tau[row, col])
                tau[row, col] -= first[row]
            a = a * np.exp(-2j * np.pi * fc * tau)[:, None, None, :]
            # A single channel with very many paths is summed in blocks of paths
            block = max(1, _CFR_MAX_ELEMENTS // (n * grid))
            for p0 in range(0, p_max, block):
                p1 = min(p0 + block, p_max)
                phase = (2 * np.pi * nu[:, p0:p1, None, None] * times[None, None, :, None]
                         - 2 * np.pi * freqs[None, None, None, :] * tau[:, p0:p1, None, None])
                h_chunk += np.einsum("rqep,rptf->rqetf", a[..., p0:p1], np.exp(1j * phase))
        if normalize:
            c = np.mean(np.abs(h_chunk) ** 2, axis=(1, 2, 3, 4))
            scale = np.where(c > 0, 1 / np.sqrt(np.where(c > 0, c, 1)), 0)
            h_chunk = h_chunk * scale[:, None, None, None, None]
        h[start:stop, :, 0] = h_chunk
        start = stop
    return h


def planar_array_positions(layout: Dict, wavelength: float) -> np.ndarray:
    """
    Element positions (m) of a planar array in its local frame, [num_rows * num_cols, 3].

    As Sionna's PlanarArray: the array lies in the y-z plane, centred at the
    origin, and the elements are numbered column-first from the top, so the
    vertical index runs fastest.
    """
    rows, cols = int(layout["num_rows"]), int(layout["num_cols"])
    d_v, d_h = float(layout["vertical_spacing"]), float(layout["horizontal_spacing"])
    r = np.tile(np.arange(rows), cols)
    c = np.repeat(np.arange(cols), rows)
    pos = np.zeros((rows * cols, 3))
    pos[:, 1] = d_h * c - (cols - 1) * d_h / 2
    pos[:, 2] = -d_v * r + (rows - 1) * d_v / 2
    return pos * wavelength


def rotation_matrix(angles) -> np.ndarray:
    """Rotation matrix of Sionna's orientation angles (rotations about z, y and x; TR 38.901 eq. 7.1-4)."""
    a, b, c = (float(x) for x in np.asarray(angles, dtype=np.float64).reshape(3))
    sa, ca, sb, cb, sc, cc = np.sin(a), np.cos(a), np.sin(b), np.cos(b), np.sin(c), np.cos(c)
    return np.array([[ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
                     [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
                     [-sb, cb * sc, cb * cc]])


def _unit_vectors(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    theta, phi = theta.astype(np.float64), phi.astype(np.float64)
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


def _array_phases(layout: Dict, wavelength: float, orientations: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Plane-wave phase of each element for each path, [P, num_elements].

    orientations: [P, 3] orientation of the array that the path leaves or
    reaches; k: [P, 3] direction of departure or arrival.
    """
    pos = planar_array_positions(layout, wavelength)
    out = np.empty((len(k), len(pos)))
    for o in np.unique(orientations, axis=0):
        sel = np.all(orientations == o, axis=1)
        rotated = pos @ rotation_matrix(o).T
        out[sel] = 2 * np.pi / wavelength * (k[sel] @ rotated.T)
    return out


def retarget_arrays(path_data: Dict[str, np.ndarray], metadata: Dict,
                    tx_array: Optional[Dict] = None, rx_array: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """
    Path data for other planar arrays at the same positions and orientations.

    With a synthetic array, the coefficient of each antenna pair is the
    coefficient at the array centres times a plane-wave phase per element.
    This recovers the centre coefficient of each path and polarization pair
    from the saved paths_a and applies the phases of the new arrays.
    tx_array and rx_array give any of num_rows, num_cols, vertical_spacing
    and horizontal_spacing (in wavelengths); the other settings stay those of
    the run. The element pattern, the polarization and the array orientation
    cannot change: the saved coefficients already include them.
    """
    for new in (tx_array, rx_array):
        if new and set(new) - set(_ARRAY_KEYS):
            raise ValueError(f"Only {_ARRAY_KEYS} can change, got {sorted(new)}")
    wavelength = _SPEED_OF_LIGHT / float(metadata["cfr_params"]["carrier_frequency"])
    old_tx, old_rx = metadata["tx_array"], metadata["rx_array"]
    new_tx = {**old_tx, **(tx_array or {})}
    new_rx = {**old_rx, **(rx_array or {})}
    offsets = path_data["paths_offsets"]
    num_paths = int(offsets[-1])
    # Orientation of the TX and of the RX of each path
    rows = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    tx_or = np.repeat(np.asarray(metadata["tx_orientation"], dtype=np.float64).reshape(1, 3), num_paths, axis=0)
    rx_or = np.asarray(metadata["rx_orientations"], dtype=np.float64).reshape(-1, 3)[rows]
    k_tx = _unit_vectors(path_data["paths_theta_t"], path_data["paths_phi_t"])
    k_rx = _unit_vectors(path_data["paths_theta_r"], path_data["paths_phi_r"])

    n_rp, n_tp = int(old_rx["num_polarizations"]), int(old_tx["num_polarizations"])
    e_rx, e_tx = int(old_rx["num_ant"]) // n_rp, int(old_tx["num_ant"]) // n_tp
    a = path_data["paths_a"].astype(np.complex128).reshape(num_paths, n_rp, e_rx, n_tp, e_tx)
    ph = (_array_phases(old_rx, wavelength, rx_or, k_rx)[:, None, :, None, None]
          + _array_phases(old_tx, wavelength, tx_or, k_tx)[:, None, None, None, :])
    # Centre coefficient per path and polarization pair, averaged over the elements
    a_c = np.mean(a * np.exp(-1j * ph), axis=(2, 4))
    ph_new = (_array_phases(new_rx, wavelength, rx_or, k_rx)[:, None, :, None, None]
              + _array_phases(new_tx, wavelength, tx_or, k_tx)[:, None, None, None, :])
    a_new = a_c[:, :, None, :, None] * np.exp(1j * ph_new)
    out = dict(path_data)
    out["paths_a"] = a_new.reshape(num_paths, n_rp * ph_new.shape[2], n_tp * ph_new.shape[4]).astype(np.complex64)
    return out


def retarget_velocities(path_data: Dict[str, np.ndarray], metadata: Dict,
                        rx_velocities: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Path data for other RX velocities at the same positions.

    The Doppler shift of a path changes by (v_new - v_old) . k_r / lambda,
    with k_r the direction of arrival. The paths themselves do not depend on
    the velocity, so the result is the path data of a new solve with these
    velocities. rx_velocities is one velocity [3] for all RXs or one per
    channel [num_channels, 3], in m/s.
    """
    offsets = path_data["paths_offsets"]
    num_ch = len(offsets) - 1
    v_new = np.asarray(rx_velocities, dtype=np.float64)
    if v_new.shape == (3,):
        v_new = np.repeat(v_new[None], num_ch, axis=0)
    if v_new.shape != (num_ch, 3):
        raise ValueError(f"rx_velocities must have shape [3] or [{num_ch}, 3], got {list(v_new.shape)}")
    wavelength = _SPEED_OF_LIGHT / float(metadata["cfr_params"]["carrier_frequency"])
    v_old = np.asarray(metadata["rx_velocities"], dtype=np.float64).reshape(num_ch, 3)
    rows = np.repeat(np.arange(num_ch), np.diff(offsets))
    k_rx = _unit_vectors(path_data["paths_theta_r"], path_data["paths_phi_r"])
    shift = np.sum((v_new - v_old)[rows] * k_rx, axis=1) / wavelength
    out = dict(path_data)
    out["paths_doppler"] = (path_data["paths_doppler"].astype(np.float64) + shift).astype(np.float32)
    return out


def _weighted_rms(x: np.ndarray, w: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w))


def path_stats(path_data: Dict[str, np.ndarray], tx_position: np.ndarray,
               rx_positions: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Per-channel statistics of the paths (see PATH_STATS).

    Parameters
    ----------
    path_data : dict
        Output of extract_path_data / concat_path_data.
    tx_position : np.ndarray
        TX position, 3 values.
    rx_positions : np.ndarray
        RX positions aligned with the channels, [num_channels, 3] or [num_channels, 3, 1].
    """
    offsets = path_data["paths_offsets"]
    num_ch = len(offsets) - 1
    out = {k: np.full(num_ch, np.nan, dtype=np.float32) for k in PATH_STATS}
    out["num_paths"] = np.diff(offsets).astype(np.int32)
    rx = np.asarray(rx_positions, dtype=np.float64).reshape(num_ch, 3)
    out["distance_m"] = np.linalg.norm(rx - np.asarray(tx_position, dtype=np.float64).reshape(1, 3),
                                       axis=1).astype(np.float32)
    power_all = _path_power(path_data)
    inter = path_data["paths_interactions"]
    los_all = np.all(inter == INTERACTION_NONE, axis=1)
    bounces_all = np.sum(inter != INTERACTION_NONE, axis=1)
    has_type = {name: np.any(inter == code, axis=1) for name, code in INTERACTION_TYPES.items()}

    for i in range(num_ch):
        s = slice(offsets[i], offsets[i + 1])
        p = power_all[s]
        if len(p) == 0 or p.sum() <= 0:
            continue
        tau = path_data["paths_tau"][s].astype(np.float64)
        total = p.sum()
        out["path_gain_db"][i] = 10 * np.log10(total)
        out["first_delay_s"][i] = tau.min()
        mean_tau = np.sum(p * tau) / total
        out["mean_delay_s"][i] = mean_tau
        out["delay_spread_s"][i] = _weighted_rms(tau, p, mean_tau)
        los = los_all[s]
        if los.any():
            p_los = p[los].sum()
            rest = total - p_los
            out["k_factor_db"][i] = 10 * np.log10(p_los / rest) if rest > 0 else np.inf
        for key, name in (("asd_rad", "paths_phi_t"), ("asa_rad", "paths_phi_r")):
            phi = path_data[name][s].astype(np.float64)
            mu = np.angle(np.sum(p * np.exp(1j * phi)))
            dphi = np.angle(np.exp(1j * (phi - mu)))
            out[key][i] = np.sqrt(np.sum(p * dphi ** 2) / total)
        for key, name in (("zsd_rad", "paths_theta_t"), ("zsa_rad", "paths_theta_r")):
            theta = path_data[name][s].astype(np.float64)
            out[key][i] = _weighted_rms(theta, p, np.sum(p * theta) / total)
        # Paths are sorted by power, so the first one is the strongest
        out["dominant_power_fraction"][i] = p[0] / total
        out["dominant_tau_s"][i] = tau[0]
        for key, name in (("dominant_theta_t", "paths_theta_t"), ("dominant_phi_t", "paths_phi_t"),
                          ("dominant_theta_r", "paths_theta_r"), ("dominant_phi_r", "paths_phi_r"),
                          ("dominant_doppler_hz", "paths_doppler")):
            out[key][i] = path_data[name][s][0]
        out["max_abs_doppler_hz"][i] = np.abs(path_data["paths_doppler"][s]).max()
        out["dominant_num_bounces"][i] = bounces_all[s][0]
        out["max_num_bounces"][i] = bounces_all[s].max()
        out["power_frac_los"][i] = p[los].sum() / total
        for name, has in has_type.items():
            out[f"power_frac_{name}"][i] = p[has[s]].sum() / total
    return out


def load_cfr(npz_path, num_subcarriers: Optional[int] = None, num_ofdm_symbols: Optional[int] = None,
             subcarrier_spacing: Optional[float] = None, normalize_delays: Optional[bool] = None,
             normalize: Optional[bool] = None, symbol_duration: Optional[float] = None,
             tx_array: Optional[Dict] = None, rx_array: Optional[Dict] = None,
             rx_velocities: Optional[np.ndarray] = None, max_paths: Optional[int] = None,
             power_range_db: Optional[float] = None, max_bounces: Optional[int] = None,
             exclude_interactions: Optional[List[str]] = None) -> np.ndarray:
    """
    Rebuild the CFR of a channel_tx_<i>.npz file from its saved paths.

    Without arguments, this returns the same array as the file's ``h`` when
    it was saved. Any argument overrides the numerology or the normalization
    of the run, e.g. more subcarriers or symbols, or a symbol duration with a
    cyclic prefix, without new ray tracing. tx_array and rx_array change the
    size or spacing of the planar arrays (see retarget_arrays);
    rx_velocities changes the RX velocities (see retarget_velocities).
    max_paths, power_range_db, max_bounces and exclude_interactions build
    the CFR from only some of the paths (see select_paths).
    """
    with np.load(Path(npz_path), allow_pickle=True) as z:
        meta = z["metadata"].item()
        if "paths_offsets" not in z:
            raise ValueError(f"{npz_path} has no saved paths (save_paths was false)")
        path_data = {key: z[key] for key in PATH_KEYS}
    if any(x is not None for x in (max_paths, power_range_db, max_bounces, exclude_interactions)):
        path_data = select_paths(path_data, max_paths, power_range_db, max_bounces, exclude_interactions)
    if tx_array or rx_array:
        path_data = retarget_arrays(path_data, meta, tx_array, rx_array)
    if rx_velocities is not None:
        path_data = retarget_velocities(path_data, meta, rx_velocities)
    p = meta["cfr_params"]
    return cfr_from_paths(
        path_data,
        carrier_frequency=p["carrier_frequency"],
        num_subcarriers=p["num_subcarriers"] if num_subcarriers is None else num_subcarriers,
        num_ofdm_symbols=p["num_ofdm_symbols"] if num_ofdm_symbols is None else num_ofdm_symbols,
        subcarrier_spacing=p["subcarrier_spacing"] if subcarrier_spacing is None else subcarrier_spacing,
        normalize_delays=p["normalize_delays"] if normalize_delays is None else normalize_delays,
        normalize=p["normalize"] if normalize is None else normalize,
        symbol_duration=symbol_duration,
    )
