import numpy as np
import random


from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from CustomPandaPickAndPlaceEnv.custom_env.env.task.PandaPickAndPlaceAndThrowTask import PandaPickAndPlaceThrowTask
from panda_gym.pybullet import PyBullet
from typing import Any, Dict, Optional, Tuple, Union
from panda_gym.utils import distance


class PandaPickAndPlaceAndThrowEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.
    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PandaPickAndPlaceThrowTask(sim, reward_type="dense")
        super().__init__(robot, task)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.task.take_step()
        self.robot.set_action(action)
        self.sim.step()
        obs = self._get_obs()
        done = False
        info = {"is_success": self.task.is_success(obs["achieved_goal"], self.task.get_goal(),),
                "gripper_distance": distance(self.robot.get_ee_position(), self.task.get_goal())}
        reward = self.task.compute_reward(
            obs["achieved_goal"],
            self.task.get_goal(),
            info)
        assert isinstance(reward, float)  # needed for pytype cheking
        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        self.task.moving_direction = random.randint(0, 1)
        return super(PandaPickAndPlaceAndThrowEnv, self).reset()
