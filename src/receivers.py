"""
Receiver creation utilities for adding user equipment to scenes.
"""

from pathlib import Path
from sionna.rt import Receiver
import numpy as np
from typing import Any, Dict, List, Tuple
import logging

from src.user_equipment import generate_ue_parameters
from src.utils import get_tx_color, scene_digest

logger = logging.getLogger(__name__)

# Tolerances for matching the base stations of an earlier run (receivers_from): position in meters, orientation in rad
BS_POSITION_TOLERANCE = 1e-3
BS_ORIENTATION_TOLERANCE = 1e-5
# Tolerance for the same user in several files of an earlier run (position m, velocity m/s, orientation rad)
USER_TOLERANCE = 1e-4
# Metadata keys a run must have for its receivers to be replayed (CSIGen 0.3.3 and later, not a split)
REPLAY_KEYS = ("tx_idx", "tx_position", "tx_orientation", "rx_names", "rx_positions", "rx_velocities",
               "rx_orientations", "num_txs", "users_per_tx", "total_users", "num_sectors", "config")
# Config keys that set the scene and the base stations, which must equal the earlier run's
REPLAY_SCENE_KEYS = ("scene_center", "antenna_height_offset", "num_deployment_buildings", "bs_positions",
                     "bs_mast_height", "clip_terrain_to_buildings", "terrain_clip_margin", "user_shift_from_ground",
                     "override_ground_material", "num_sectors", "mechanical_tilt", "azimuth_offset")
# Run-level metadata that every file of a run shares
REPLAY_RUN_KEYS = ("num_txs", "users_per_tx", "total_users", "num_sectors")


def rx_names_for_tx(users_per_tx: List[int], tx_idx: int) -> List[str]:
    """
    Names of the users sampled for TX ``tx_idx``.

    Users are named UE_<k> with k counting TX by TX (see
    add_receivers_from_samples), so the users of TX t are
    UE_<sum(users_per_tx[:t])> to UE_<sum(users_per_tx[:t+1]) - 1>.
    """
    start = int(sum(users_per_tx[:tx_idx]))
    return [f"UE_{k}" for k in range(start, start + int(users_per_tx[tx_idx]))]


def add_receivers_from_samples(
    scene,
    sampled_positions: Tuple[List[np.ndarray], List[np.ndarray]],
    num_sectors: int,
    mobility_preset: str,
    mobility_presets: dict,
    seed: int = 1
) -> Tuple[int, List[int], int]:
    """
    Add receivers to scene from sampled positions.
    
    Parameters
    ----------
    scene : sionna.rt.Scene
        The Sionna scene object
    sampled_positions : tuple
        (positions, cell_ids) from sample_user_positions(): lists with one
        entry per TX, of shapes [num_users_tx, 3] and [num_users_tx]
    num_sectors : int
        Number of sectors per base station
    mobility_preset : str
        Key for mobility preset in mobility_presets dict
    mobility_presets : dict
        Dictionary of mobility presets
    seed : int, default=1
        Seed for UE parameter generation
    
    Returns
    -------
    tuple
        (num_txs, users_per_tx, total_users), where users_per_tx lists the
        number of users of each TX. Users are named UE_<k> with k counting
        TX by TX; rx_names_for_tx() gives the names of the users of a TX.
    """
    # Extract positions and cell_ids
    positions, _ = sampled_positions
    num_txs = len(positions)
    users_per_tx = [len(p) for p in positions]
    total_users = sum(users_per_tx)
    if total_users == 0:
        logger.warning(f"Total TXs: {num_txs}, no users to add")
    else:
        logger.info(f"Total TXs: {num_txs}, users per TX: {min(users_per_tx)}-{max(users_per_tx)}, total users: {total_users}")
    
    # Get config from selected preset and generate UE parameters
    if mobility_preset not in mobility_presets:
        raise ValueError(f"Mobility preset '{mobility_preset}' not found in mobility_presets")
    preset_config = mobility_presets[mobility_preset]
    orientation_mode = preset_config.get("orientation_mode", "random")
    
    orientations, velocities = generate_ue_parameters(
        num_ues=total_users,
        seed=seed,
        **preset_config
    )
    
    # Add receivers for each user
    user_count = 0
    for tx_idx in range(num_txs):
        # Map tx_idx to TX name: BS_{bs_id}_sector_{sector_id}
        bs_id = tx_idx // num_sectors
        sector_id = (tx_idx % num_sectors) + 1
        tx_name = f"BS_{bs_id}_sector_{sector_id}"
        
        # Get the TX object from the scene (needed for look_at mode)
        tx_object = scene.get(tx_name)
        
        # Color for this TX's users (for visualization purposes)
        color = get_tx_color(tx_idx, num_txs)
        
        for user_idx in range(users_per_tx[tx_idx]):
            pos = positions[tx_idx][user_idx].tolist()  # [x, y, z]
            vel = velocities[user_count].tolist()       # [vx, vy, vz]
            
            if orientation_mode == "random":
                rx = Receiver(
                    name=f"UE_{user_count}",
                    position=pos,
                    orientation=orientations[user_count].tolist(),
                    velocity=vel,
                    color=color
                )
            else:  # "to_tx"
                rx = Receiver(
                    name=f"UE_{user_count}",
                    position=pos,
                    look_at=tx_object,
                    velocity=vel,
                    color=color
                )
            
            scene.add(rx)
            user_count += 1
    
    logger.info(f"Added {user_count} receivers to scene")
    logger.info(f"  - Preset: {mobility_preset}")
    logger.info(f"  - Orientation: {orientation_mode}, Speed: {preset_config.get('speed_distribution')}")
    
    return num_txs, users_per_tx, total_users


def load_receivers(run_dir) -> Dict[str, Any]:
    """
    Receivers and base stations of an earlier run, for solving them again (receivers_from).

    Reads the metadata of every channel_tx_<t>.npz in ``run_dir``. A run saves
    each user in the files of the TXs it has at least one path to, with its
    position, velocity and orientation; users with no path to any TX are not
    in the files and cannot be replayed.

    Returns
    -------
    dict
        'users_per_tx', 'total_users', 'num_txs', 'num_sectors' of the run;
        'users': {k: (position [3], velocity [3], orientation [3])} for the
        users UE_k found in the files; 'tx': {t: (position [3], orientation [3])};
        'config' and 'scene_sha256' (None before CSIGen 0.3.6) of the run.
    """
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("channel_tx_*.npz"), key=lambda f: int(f.stem.split("_")[-1]))
    if not files:
        raise ValueError(f"No channel_tx_*.npz files in {run_dir}")
    users, tx, first = {}, {}, None
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            m = z["metadata"].item()
        missing = [k for k in REPLAY_KEYS if k not in m]
        if missing:
            raise ValueError(f"{f} lacks {missing}: receivers_from needs a full run of CSIGen 0.3.3 or later "
                             f"(not the output of split_dataset.py)")
        first = first or m
        for k in REPLAY_RUN_KEYS:
            if list(np.atleast_1d(m[k])) != list(np.atleast_1d(first[k])):
                raise ValueError(f"{f} has another {k} than {files[0]}: the files are not from one run")
        if m.get("scene_sha256") != first.get("scene_sha256"):
            raise ValueError(f"{f} has another scene_sha256 than {files[0]}: the files are not from one run")
        tx[int(m["tx_idx"])] = (np.asarray(m["tx_position"], dtype=np.float64).reshape(3),
                                np.asarray(m["tx_orientation"], dtype=np.float64).reshape(3))
        ks = np.array([int(name.split("_")[1]) for name in m["rx_names"]], dtype=np.int64)
        state = np.concatenate([np.asarray(m[key], dtype=np.float64).reshape(len(ks), 3)
                                for key in ("rx_positions", "rx_velocities", "rx_orientations")], axis=1)   # [n, 9]
        seen = np.array([k in users for k in ks], dtype=bool)
        if seen.any():                              # the same users in several files must agree (one comparison)
            ref = np.stack([np.concatenate(users[k]) for k in ks[seen]])
            if not np.allclose(ref, state[seen], atol=USER_TOLERANCE):
                bad = ks[seen][~np.isclose(ref, state[seen], atol=USER_TOLERANCE).all(1)][0]
                raise ValueError(f"UE_{bad} has different positions, velocities or orientations in the files of "
                                 f"{run_dir}")
        for k, row in zip(ks[~seen], state[~seen]):
            users[int(k)] = (row[0:3], row[3:6], row[6:9])
    num_txs = int(first["num_txs"])
    if sorted(tx) != list(range(num_txs)):
        raise ValueError(f"{run_dir}: expected channel_tx_0..{num_txs - 1}.npz, found TXs {sorted(tx)}")
    out = {
        "users_per_tx": [int(n) for n in first["users_per_tx"]],
        "total_users": int(first["total_users"]),
        "num_txs": num_txs,
        "num_sectors": int(first["num_sectors"]),
        "users": users,
        "tx": tx,
        "config": first["config"],
        "scene_sha256": first.get("scene_sha256"),      # recorded since CSIGen 0.3.6
    }
    logger.info("Receivers from %s (%.2f GHz): %d of %d users (the others have no path to any TX there), %d TXs",
                run_dir, float(first["config"]["carrier_frequency"]) / 1e9, len(users), out["total_users"], num_txs)
    return out


def check_replay_base_stations(scene, replay: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Raise if the scene or the base stations differ from those of the run whose
    receivers are replayed: the same scene contents (scene_digest), scene-setup
    and placement keys, and the same TXs (no more, no fewer) at the same
    positions and orientations.
    """
    old_cfg = replay["config"]
    for key in REPLAY_SCENE_KEYS:
        if config.get(key) != old_cfg.get(key):
            raise ValueError(f"{key} is {config.get(key)!r}, in the replayed run {old_cfg.get(key)!r}")
    # The scene by its contents (scene_digest): the earlier run's recorded digest, or the digest of its scene file
    # if its path still resolves; else only the scene's directory and file name can be compared.
    new_digest = scene_digest(config["scene_xml_path"])
    old_scene = Path(old_cfg["scene_xml_path"])
    old_digest = replay.get("scene_sha256") or (scene_digest(old_scene) if old_scene.is_file() else None)
    if old_digest is not None:
        if old_digest != new_digest:
            raise ValueError(f"The scene {config['scene_xml_path']} differs from the replayed run's ({old_scene}): "
                             f"other geometry or materials")
    else:
        new_scene = Path(config["scene_xml_path"])
        if (old_scene.parent.name, old_scene.name) != (new_scene.parent.name, new_scene.name):
            raise ValueError(f"scene_xml_path is {new_scene}, in the replayed run {old_scene}")
        logger.warning("The replayed run records no scene digest and its scene file %s is not found: only the scene's "
                       "name is compared", old_scene)
    num_sectors = int(config["num_sectors"])
    names = {f"BS_{t // num_sectors}_sector_{t % num_sectors + 1}": t for t in replay["tx"]}
    extra = sorted(tx for tx in scene.transmitters if tx not in names)
    if extra:
        raise ValueError(f"The scene has TXs that the replayed run does not: {extra}")
    for name, t in names.items():
        pos_ref, ori_ref = replay["tx"][t]
        tx = scene.transmitters.get(name)
        if tx is None:
            raise ValueError(f"The replayed run has {name}, the scene does not")
        pos = np.asarray(tx.position.numpy(), dtype=np.float64).reshape(3)
        ori = np.asarray(tx.orientation.numpy(), dtype=np.float64).reshape(3)
        if not np.allclose(pos, pos_ref, atol=BS_POSITION_TOLERANCE):
            raise ValueError(f"{name} is at {pos.tolist()}, in the replayed run at {pos_ref.tolist()}")
        if not np.allclose(ori, ori_ref, atol=BS_ORIENTATION_TOLERANCE):
            raise ValueError(f"{name} has orientation {ori.tolist()}, in the replayed run {ori_ref.tolist()}")


def add_receivers_from_run(scene, replay: Dict[str, Any]) -> List[str]:
    """
    Add the receivers of an earlier run (load_receivers) to the scene, with
    their names, positions, velocities and orientations, so that the users keep
    their indices and serving TXs. Returns the names added, in index order.
    """
    num_txs = replay["num_txs"]
    serving = np.repeat(np.arange(num_txs), replay["users_per_tx"])
    names = []
    for k in sorted(replay["users"]):
        pos, vel, ori = replay["users"][k]
        scene.add(Receiver(name=f"UE_{k}", position=pos.tolist(), orientation=ori.tolist(), velocity=vel.tolist(),
                           color=get_tx_color(int(serving[k]), num_txs)))
        names.append(f"UE_{k}")
    logger.info("Added %d receivers of the replayed run to the scene", len(names))
    return names
