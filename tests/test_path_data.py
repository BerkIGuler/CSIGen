from pathlib import Path

import numpy as np
import pytest

from helpers import CFR, NUM_RX_ANT, NUM_TX_ANT, RX_ARRAY, TX_ARRAY, random_paths
from src import path_data
from src.path_data import (PATH_KEYS, cfr_from_paths, load_cfr, path_stats, retarget_arrays, retarget_velocities,
                           select_paths, slice_path_data, subcarrier_offsets)

FC, K, T, SCS = CFR["carrier_frequency"], CFR["num_subcarriers"], CFR["num_ofdm_symbols"], CFR["subcarrier_spacing"]
ALL = {**CFR, "normalize_delays": True, "normalize": True}
GOLDEN = Path(__file__).parent / "data" / "golden_cfr.npz"


def reference_cfr(pd, i, symbol_duration=1 / SCS):
    """h(f, t) = sum_l a_l exp(-j 2 pi (f_c + f) tau_l) exp(j 2 pi nu_l t) of channel i, in float64."""
    s, e = pd["paths_offsets"][i], pd["paths_offsets"][i + 1]
    a = pd["paths_a"][s:e].astype(np.complex128)
    tau = pd["paths_tau"][s:e].astype(np.float64)[:, None, None]
    nu = pd["paths_doppler"][s:e].astype(np.float64)[:, None, None]
    f = subcarrier_offsets(K, SCS)[None, None, :]
    t = (np.arange(T) * symbol_duration)[None, :, None]
    phase = np.exp(-2j * np.pi * (FC + f) * tau) * np.exp(2j * np.pi * nu * t)
    return np.einsum("pqe,ptf->qetf", a, phase)


def test_subcarriers_are_centered_on_the_carrier():
    assert np.allclose(subcarrier_offsets(4, 1.0), [-2, -1, 0, 1])
    assert np.allclose(subcarrier_offsets(5, 1.0), [-2, -1, 0, 1, 2])


@pytest.mark.parametrize("symbol_duration", [None, 1.07 / SCS])
def test_cfr_is_the_sum_over_paths(symbol_duration):
    pd = random_paths([3, 0, 5, 1])
    h = cfr_from_paths(pd, **CFR, symbol_duration=symbol_duration)
    assert h.shape == (4, NUM_RX_ANT, 1, NUM_TX_ANT, T, K) and h.dtype == np.complex64
    for i in range(4):
        ref = reference_cfr(pd, i, 1 / SCS if symbol_duration is None else symbol_duration)
        assert np.allclose(h[i, :, 0], ref, rtol=1e-5, atol=1e-12)
    assert not h[1].any()                           # a channel without paths is zero


def test_normalizations():
    pd = random_paths([3, 0, 2])
    h = cfr_from_paths(pd, **{**CFR, "normalize": True})
    energy = np.mean(np.abs(h) ** 2, axis=(1, 2, 3, 4, 5))
    assert np.allclose(energy[[0, 2]], 1.0, rtol=1e-5) and energy[1] == 0      # a channel without paths stays zero
    # Delays relative to the first path of each channel, in float64 as cfr_from_paths subtracts them
    shifted = dict(pd, paths_tau=pd["paths_tau"].astype(np.float64))
    for s, e in zip(pd["paths_offsets"][:-1], pd["paths_offsets"][1:]):
        if e > s:
            shifted["paths_tau"][s:e] -= shifted["paths_tau"][s:e].min()
    h = cfr_from_paths(pd, **{**CFR, "normalize_delays": True})
    assert np.all(np.isfinite(h)) and not h[1].any()
    assert np.allclose(h, cfr_from_paths(shifted, **CFR), rtol=1e-5, atol=1e-12)
    assert np.all(np.isfinite(cfr_from_paths(pd, **ALL)))


@pytest.fixture
def chunk_paths():
    return random_paths([4, 7, 1, 0, 6, 3])


def test_chunks_do_not_change_the_bits(chunk_paths, monkeypatch):
    # The chunks of several channels only bound the memory, and each channel is summed the same way
    whole = cfr_from_paths(chunk_paths, **ALL)
    monkeypatch.setattr(path_data, "_CFR_CHUNK_ELEMENTS", 1)        # one channel per chunk, one block of paths
    assert cfr_from_paths(chunk_paths, **ALL).tobytes() == whole.tobytes()


def test_path_bound_applies_only_to_a_single_channel(chunk_paths, monkeypatch):
    # Blocks of paths give the same bits here, so count the path sums. A chunk of several channels
    # is summed in one block, whatever the path bound
    calls = []
    einsum = np.einsum
    monkeypatch.setattr(np, "einsum", lambda *a, **k: calls.append(a[0]) or einsum(*a, **k))
    whole = cfr_from_paths(chunk_paths, **ALL)
    unbounded = len(calls)
    calls.clear()
    monkeypatch.setattr(path_data, "_CFR_MAX_ELEMENTS", 1)
    assert cfr_from_paths(chunk_paths, **ALL).tobytes() == whole.tobytes()
    assert len(calls) == unbounded


def test_blocks_of_paths_keep_the_cfr(chunk_paths, monkeypatch):
    # A single channel above the bound sums its paths in blocks, here of one path. This can change the last bits
    whole = cfr_from_paths(chunk_paths, **ALL)
    monkeypatch.setattr(path_data, "_CFR_CHUNK_ELEMENTS", 1)
    monkeypatch.setattr(path_data, "_CFR_MAX_ELEMENTS", 1)
    assert np.allclose(cfr_from_paths(chunk_paths, **ALL), whole, rtol=1e-5, atol=1e-6)


def test_load_cfr_rebuilds_the_saved_cfr(run_dir):
    f = run_dir / "channel_tx_0.npz"
    with np.load(f, allow_pickle=True) as z:
        h = z["h"]
        pd = {key: z[key] for key in PATH_KEYS}
    rebuilt = load_cfr(f)
    assert rebuilt.tobytes() == h.tobytes()
    for i in range(len(h)):
        assert np.allclose(rebuilt[i, :, 0], reference_cfr(pd, i), rtol=1e-5, atol=1e-12)
    more = load_cfr(f, num_subcarriers=16)
    assert more.shape[-1] == 16
    assert more.tobytes() == cfr_from_paths(pd, **{**CFR, "num_subcarriers": 16}).tobytes()


def make_golden(path=GOLDEN):
    """
    Write the golden file. It was made once with CSIGen 0.4.2 (commit b07b2ed) and is not made again by the tests.
    It holds 4 channels with 2, 0, 3 and 1 paths, 2 x 3 antennas and a grid of 2 symbols and 4 subcarriers, with
    both normalizations on.
    """
    pd = random_paths([2, 0, 3, 1], seed=7)
    params = {**ALL, "num_subcarriers": 4, "num_ofdm_symbols": 2}
    h = cfr_from_paths(pd, **params)
    meta = {"cfr_params": params, "cfr_shape": h.shape, "cfr_source": "cfr_from_paths", "paths_saved": True,
            "csigen_version": "0.4.2"}
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, h=h, metadata=np.array(meta, dtype=object), **pd)


def test_load_cfr_gives_the_golden_cfr():
    """
    load_cfr gives the saved h within a few float32 steps of the largest value. The test does not compare the bits,
    because np.exp and np.einsum can round differently on another CPU or with another NumPy build. A change of the
    formula, the axes or the delays of cfr_from_paths is much larger than this tolerance.
    """
    with np.load(GOLDEN, allow_pickle=True) as z:
        h = z["h"]
    assert h.shape == (4, NUM_RX_ANT, 1, NUM_TX_ANT, 2, 4)
    tol = 4 * np.finfo(np.float32).eps * np.abs(h).max()
    assert np.allclose(load_cfr(GOLDEN), h, rtol=0, atol=tol)
    rebuilt = load_cfr(GOLDEN)
    assert np.allclose(np.mean(np.abs(rebuilt[[0, 2, 3]]) ** 2, axis=(1, 2, 3, 4, 5)), 1.0, rtol=1e-5)


def file_paths(f):
    with np.load(f, allow_pickle=True) as z:
        return {key: z[key] for key in PATH_KEYS}, z["metadata"].item()


def test_load_cfr_selects_paths(run_dir):
    f = run_dir / "channel_tx_0.npz"
    pd, _ = file_paths(f)
    for args in ({"max_paths": 1}, {"max_bounces": 0}):
        assert load_cfr(f, **args).tobytes() == cfr_from_paths(select_paths(pd, **args), **CFR).tobytes()


def test_retarget_to_the_same_arrays(run_dir):
    f = run_dir / "channel_tx_0.npz"
    pd, meta = file_paths(f)
    same = retarget_arrays(pd, meta, {"num_cols": NUM_TX_ANT}, {"num_cols": NUM_RX_ANT})
    assert np.allclose(same["paths_a"], pd["paths_a"], rtol=1e-5, atol=1e-11)
    assert np.allclose(load_cfr(f, tx_array={"num_cols": NUM_TX_ANT}), load_cfr(f), rtol=1e-4, atol=1e-10)


def test_retarget_to_other_arrays(run_dir):
    pd, meta = file_paths(run_dir / "channel_tx_0.npz")
    big = retarget_arrays(pd, meta, {"num_rows": 2, "num_cols": 4})
    assert big["paths_a"].shape == (len(pd["paths_tau"]), NUM_RX_ANT, 8)
    # Each path keeps its power, and the elements of the old array keep their coefficients after a way back
    assert np.allclose(np.abs(big["paths_a"][:, :, 0]), np.abs(pd["paths_a"][:, :, 0]), rtol=1e-5)
    back = retarget_arrays(big, {**meta, "tx_array": {**TX_ARRAY, "num_rows": 2, "num_cols": 4, "num_ant": 8}},
                           {"num_rows": 1, "num_cols": NUM_TX_ANT})
    assert np.allclose(back["paths_a"], pd["paths_a"], rtol=1e-5, atol=1e-11)
    with pytest.raises(ValueError, match="can change"):
        retarget_arrays(pd, meta, {"pattern": "tr38901"})


def test_retarget_velocities(run_dir):
    pd, meta = file_paths(run_dir / "channel_tx_0.npz")
    v = np.asarray(meta["rx_velocities"])
    assert np.array_equal(retarget_velocities(pd, meta, v)["paths_doppler"], pd["paths_doppler"])
    # 10 m/s more along x shifts the Doppler of a path by 10 sin(theta_r) cos(phi_r) / wavelength
    moved = retarget_velocities(pd, meta, v + [10.0, 0.0, 0.0])
    k_x = np.sin(pd["paths_theta_r"].astype(np.float64)) * np.cos(pd["paths_phi_r"].astype(np.float64))
    shift = 10.0 * k_x * CFR["carrier_frequency"] / 299792458.0
    assert np.allclose(moved["paths_doppler"], pd["paths_doppler"] + shift, rtol=1e-6, atol=1e-3)
    with pytest.raises(ValueError, match="rx_velocities must have shape"):
        retarget_velocities(pd, meta, np.zeros(2))


def test_path_stats_of_hand_made_channels():
    # Channel 0 has one LoS path. Channel 1 has a LoS path and a specular path 3 dB weaker, 100 ns later
    a = np.zeros((3, 1, 1), dtype=np.complex64)
    a[:, 0, 0] = [1e-4, 1e-4, 1e-4 / np.sqrt(2)]
    pd = {"paths_offsets": np.array([0, 1, 3]), "paths_a": a,
          "paths_tau": np.array([1e-6, 1e-6, 1.1e-6], dtype=np.float32),
          "paths_doppler": np.array([5.0, -7.0, 3.0], dtype=np.float32),
          "paths_theta_t": np.full(3, 1.0, np.float32), "paths_phi_t": np.array([0.2, 0.2, 0.5], np.float32),
          "paths_theta_r": np.full(3, 2.0, np.float32), "paths_phi_r": np.array([1.0, 1.0, 1.4], np.float32),
          "paths_interactions": np.array([[0, 0], [0, 0], [1, 0]], dtype=np.uint8)}
    st = path_stats(pd, [0.0, 0.0, 10.0], [[30.0, 40.0, 10.0], [0.0, 0.0, 1.0]])
    assert st["num_paths"].tolist() == [1, 2]
    assert np.allclose(st["distance_m"], [50.0, 9.0])
    assert np.isinf(st["k_factor_db"][0]) and np.isclose(st["k_factor_db"][1], 10 * np.log10(2), atol=1e-4)
    assert np.allclose(st["path_gain_db"], [-80.0, 10 * np.log10(1.5e-8)], atol=1e-4)
    # Powers 2/3 and 1/3 at delays 100 ns apart give a spread of sqrt(2) / 3 times 100 ns
    assert st["delay_spread_s"][0] == 0 and np.isclose(st["delay_spread_s"][1], np.sqrt(2) / 3 * 1e-7, rtol=1e-4)
    assert np.allclose(st["power_frac_los"], [1.0, 2 / 3]) and np.allclose(st["power_frac_specular"], [0.0, 1 / 3])
    assert st["max_abs_doppler_hz"].tolist() == [5.0, 7.0] and st["max_num_bounces"].tolist() == [0, 1]


def test_load_cfr_needs_saved_paths(tmp_path):
    f = tmp_path / "channel_tx_0.npz"
    np.savez(f, h=np.zeros(1), metadata=np.array({"cfr_params": CFR, "paths_saved": False}, dtype=object))
    with pytest.raises(ValueError, match="no saved paths"):
        load_cfr(f)


def test_select_paths():
    pd = random_paths([3, 0, 4])
    los = select_paths(pd, max_bounces=0)
    assert np.all(los["paths_interactions"] == 0)
    assert np.diff(los["paths_offsets"]).tolist() == [1, 0, 1]      # the first path of each channel is LoS
    strongest = select_paths(pd, max_paths=1)
    assert np.array_equal(strongest["paths_a"], pd["paths_a"][[0, 3]])
    assert select_paths(pd, power_range_db=0.0)["paths_offsets"].tolist() == [0, 1, 1, 2]


def test_power_range_keeps_the_paths_within_the_range():
    pd = random_paths([6, 0, 7], seed=5)
    # Path k of a channel is about 4 k dB below its first path, so the order by power stays
    rank = np.arange(13) - np.repeat(pd["paths_offsets"][:-1], np.diff(pd["paths_offsets"]))
    pd["paths_a"] = (pd["paths_a"] * 10 ** (-0.2 * rank)[:, None, None]).astype(np.complex64)
    power = np.mean(np.abs(pd["paths_a"].astype(np.complex128)) ** 2, axis=(1, 2))
    expected = []
    for s, e in zip(pd["paths_offsets"][:-1], pd["paths_offsets"][1:]):
        expected += [i for i in range(s, e) if power[i] >= power[s] / 10]     # 10 dB below the strongest path
    assert 2 < len(expected) < 13                   # paths on both sides of the threshold
    kept = select_paths(pd, power_range_db=10.0)
    assert np.array_equal(kept["paths_tau"], pd["paths_tau"][expected])


def test_select_paths_by_interaction():
    pd = random_paths([5, 0, 6], seed=3)
    inter = pd["paths_interactions"]
    kept = select_paths(pd, exclude_interactions=["diffraction", "specular"])
    assert not np.isin(kept["paths_interactions"], [1, 8]).any()
    assert len(kept["paths_tau"]) == int((~np.isin(inter, [1, 8]).any(axis=1)).sum())
    with pytest.raises(ValueError, match="Unknown interaction types"):
        select_paths(pd, exclude_interactions=["scattering"])


def test_max_paths_ranks_after_the_other_filters():
    pd = random_paths([5, 0, 6], seed=3)
    keep = ~(pd["paths_interactions"] == 8).any(axis=1)             # no diffraction
    expected = []
    for s, e in zip(pd["paths_offsets"][:-1], pd["paths_offsets"][1:]):
        expected += [i for i in range(s, e) if keep[i]][:2]
    kept = select_paths(pd, max_paths=2, exclude_interactions=["diffraction"])
    assert np.array_equal(kept["paths_tau"], pd["paths_tau"][expected])


def test_slice_path_data_keeps_the_order_of_the_rows():
    pd = random_paths([2, 0, 3, 1])
    part = slice_path_data(pd, np.array([3, 0, 1]))
    assert np.diff(part["paths_offsets"]).tolist() == [1, 2, 0]
    assert np.array_equal(part["paths_tau"], pd["paths_tau"][[5, 0, 1]])
    h = cfr_from_paths(pd, **CFR)
    assert cfr_from_paths(part, **CFR).tobytes() == h[[3, 0, 1]].tobytes()
