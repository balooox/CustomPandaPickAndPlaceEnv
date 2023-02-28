from typing import Any, Dict, Union

import numpy as np
import random
import pybullet as p

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class PandaPickAndPlaceThrowTask(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([0, obj_xy_range / 2, 0])
        self.moving_direction = 1
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="moving_platform",
            half_extents=np.array([0.1, 0.1, 0.01]),
            mass=0.0,
            position=np.array([.15, 0, 0.2]),
            rgba_color=np.array([0.5, 0.2, 0.2, 1.0]),
        )

    def take_step(self):
        cur_moving_platform = self.sim.get_base_position("moving_platform")
        cur_moving_target = self.sim.get_base_position("target")
        orientation = np.array([0, 0, 0, 1])
        if self.moving_direction == 1:
            cur_moving_platform[1] += 0.003
            cur_moving_target[1] += 0.003
            self.sim.set_base_pose("moving_platform", cur_moving_platform, orientation)
            self.sim.set_base_pose("target", cur_moving_target, orientation)
        elif self.moving_direction == 0:
            cur_moving_platform[1] -= 0.003
            cur_moving_target[1] -= 0.003
            self.sim.set_base_pose("moving_platform", cur_moving_platform, orientation)
            self.sim.set_base_pose("target", cur_moving_target, orientation)

        self.goal = self.sim.get_base_position("target").copy()

        contact = p.getContactPoints(self.sim._bodies_idx["object"], self.sim._bodies_idx["moving_platform"])

        if len(contact) > 0:
            # print("Collision")
            cur_object = self.sim.get_base_position("object")
            if self.moving_direction == 1:
                cur_object[1] += 0.003
                self.sim.set_base_pose("object", cur_object, self.sim.get_base_rotation("object"))
            elif self.moving_direction == 0:
                cur_object[1] -= 0.003
                self.sim.set_base_pose("object", cur_object, self.sim.get_base_rotation("object"))
        else:
            # print("No collision")
            pass

        if cur_moving_platform[1] > 0.25:
            self.moving_direction = 0
        elif cur_moving_platform[1] < -0.25:
            self.moving_direction = 1

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object

        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        moving_platform_position = self.sim.get_base_position("object")
        moving_platform_rotation = self.sim.get_base_rotation("object")
        moving_platform_velocity = self.sim.get_base_velocity("object")
        moving_platform_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate(
            [object_position,
             object_rotation,
             object_velocity,
             object_angular_velocity,
             moving_platform_position,
             moving_platform_rotation,
             moving_platform_velocity,
             moving_platform_angular_velocity
             ])
        return observation


    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        platform_position = self._random_moving_pos()
        self.goal = self._sample_goal(platform_position)
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("moving_platform", platform_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _random_moving_pos(self) -> np.ndarray:
        """Generate random platform pos"""
        platform_position = np.array([.15, 0, 0.2])
        y_noise = random.randrange(-25, 25) / 100
        z_noise = random.randrange(-20, 8) / 100
        platform_position[1] += y_noise
        platform_position[2] += z_noise

        return platform_position

    def _sample_goal(self, platform_position: np.ndarray) -> np.ndarray:
        """Sample a goal."""
        goal_position = platform_position.copy()
        goal_position[2] += 0.04

        return goal_position

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        r_gripper = 0

        if len(info) == 2:
            if info["gripper_distance"] < 0.05:
                r_gripper = 2
        else:
            r_gripper = []
            for i in range(len(info)):
                if info[i]["gripper_distance"] < 0.05:
                    r_gripper.append(2)
                else:
                    r_gripper.append(0)

            r_gripper = np.asarray(r_gripper)

        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d - r_gripper

