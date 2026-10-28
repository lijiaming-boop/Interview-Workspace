"""Isaac Lab config derived from the official Franka cube-lift scene."""

from __future__ import annotations

from typing import Tuple

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
    from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
        FrankaCubeLiftEnvCfg,
    )
    from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import ObjectTableSceneCfg
except ImportError as exc:  # pragma: no cover - loaded only inside Isaac Lab
    raise ImportError(
        "bridge_pick_place_env_cfg.py must run inside an Isaac Lab environment. "
        "Install Isaac Lab and launch with isaaclab.sh/isaaclab.bat."
    ) from exc


BRIDGE_TARGET_XYZ: Tuple[float, float, float] = (0.68, -0.18, 0.055)
BRIDGE_CAMERA_POS = (1.05, -1.05, 0.82)
BRIDGE_CAMERA_ROT = (0.64, -0.28, 0.28, 0.66)
RAW_CAMERA_POS = (1.20, -1.20, 1.05)


@configclass
class BridgeTableSceneCfg(ObjectTableSceneCfg):
    """Official lift scene plus the camera and visual placement target."""

    place_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PlaceTarget",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(BRIDGE_TARGET_XYZ)),
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.004),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.70, 0.20), roughness=0.8
            ),
        ),
    )
    bridge_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/BridgeCamera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=BRIDGE_CAMERA_POS,
            # ROS camera convention; tune only this and pos in alignment sweeps.
            rot=BRIDGE_CAMERA_ROT,
            convention="ros",
        ),
    )


@configclass
class BridgePickPlaceEnvCfg(FrankaCubeLiftEnvCfg):
    """Single-cube tabletop pick-and-place using official Franka relative IK."""

    scene: BridgeTableSceneCfg = BridgeTableSceneCfg(num_envs=8, env_spacing=2.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.episode_length_s = 12.0
        self.decimation = 5
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        # Keep the official high-PD Panda and DLS relative IK controller.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107]
            ),
        )


@configclass
class BridgePickPlaceEnvCfg_PLAY(BridgePickPlaceEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False


def apply_camera_profile(env_cfg: BridgePickPlaceEnvCfg, mode: str) -> None:
    """Select the fixed camera pose used by a domain-gap condition."""

    if mode == "raw":
        env_cfg.scene.bridge_camera.offset.pos = RAW_CAMERA_POS
    elif mode in {"camera_aligned", "bridge_aligned"}:
        env_cfg.scene.bridge_camera.offset.pos = BRIDGE_CAMERA_POS
        env_cfg.scene.bridge_camera.offset.rot = BRIDGE_CAMERA_ROT
    else:
        raise ValueError(f"unsupported camera profile: {mode}")
