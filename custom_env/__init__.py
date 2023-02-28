import os
import sys

sys.path.append("..")

from gym.envs.registration import register

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="PandaPickAndPlaceAndThrow{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="CustomPandaPickAndPlaceEnv.custom_env.env:PandaPickAndPlaceAndThrowEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPickAndPlaceAndMove{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="CustomPandaPickAndPlaceEnv.custom_env.env:PandaPickAndPlaceAndMoveEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )
