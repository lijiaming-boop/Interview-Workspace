"""Register the custom task for gym.make inside Isaac Lab."""

import gymnasium as gym

gym.register(
    id="Isaac-Bridge-PickPlace-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "bridge_pick_place_env_cfg:BridgePickPlaceEnvCfg"
        )
    },
    disable_env_checker=True,
)
