from typing import Union, Dict, Tuple, List
import ipdb

import gym
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
import rlbench
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

import pyrep


class GraspEnv(gym.Env):
    """An gym wrapper for Team Grasp."""

    metadata = {'render.modes': ['human', 'rgb_array']}
    ee_control_types = set([
        ArmActionMode.ABS_EE_POSE_WORLD_FRAME,
        ArmActionMode.DELTA_EE_POSE_WORLD_FRAME,
        ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME,
        ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME,
        ArmActionMode.EE_POSE_EE_FRAME,
        ArmActionMode.EE_POSE_PLAN_EE_FRAME,
    ])
    delta_ee_control_types = set([
        ArmActionMode.DELTA_EE_POSE_WORLD_FRAME,
        ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME,
        ArmActionMode.EE_POSE_EE_FRAME,
        ArmActionMode.EE_POSE_PLAN_EE_FRAME
    ])

    def __init__(self, task_class, act_mode=ArmActionMode.ABS_JOINT_VELOCITY, observation_mode='state',
                 render_mode: Union[None, str] = None, epsiode_length: int = 200, action_size: Union[None, int] = None,
                 manual_terminate: bool = True, penalize_illegal: bool = True, action_range: float = 0.01):
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        self.action_range = action_range
        obs_config = ObservationConfig()
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision':
            obs_config.set_all(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)
        action_mode = ActionMode(act_mode)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self.task_class = task_class
        self.task = self.env.get_task(task_class)
        self.n_steps = 0
        self.epsiode_length = epsiode_length
        self.manual_terminate = manual_terminate
        self.penalize_illegal = penalize_illegal

        desc, obs = self.task.reset()

        print(desc)

        if action_size is None:
            action_size = self.env.action_size
        self.action_size = action_size
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_size,))

        if observation_mode == 'state':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=self.get_low_dim_data(obs).shape)
        elif observation_mode == 'vision':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "left_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                "right_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                "wrist_rgb": spaces.Box(
                    low=0, high=1, shape=obs.wrist_rgb.shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=obs.front_rgb.shape),
            })

        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def get_low_dim_data(self, obs) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """

        # low_dim_data = [] if obs.gripper_open is None else [[obs.gripper_open]]
        low_dim_data = []
        for data in [
            #  obs.joint_velocities,
            #  obs.joint_positions,
            #  obs.joint_forces,
            obs.gripper_pose[:3],
            # [obs.gripper_open],
            #  obs.gripper_joint_positions,
            #  obs.gripper_touch_forces,
            obs.task_low_dim_state,  # target state
        ]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        if self._observation_mode == 'state':
            return self.get_low_dim_data(obs)
        elif self._observation_mode == 'vision':
            return {
                "state": obs.get_low_dim_data(),
                "left_shoulder_rgb": obs.left_shoulder_rgb,
                "right_shoulder_rgb": obs.right_shoulder_rgb,
                "wrist_rgb": obs.wrist_rgb,
                "front_rgb": obs.front_rgb,
            }

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            return self._gym_cam.capture_rgb()

    # : Union[List, None]):
    def switch_task(self, task_class, target_position):
        self.task = self.env.get_task(task_class)
        self.task._task.target_position = target_position

    @staticmethod
    def switch_task_wrapper(self, task_class: rlbench.backend.task.Task,
                            target_position):  # : Union[List, None] = None):
        """Change current task by specifying desired task class. Task objects are randomly initialized.

        Args:
            task_class (rlbench.backend.task.Task): desired task class
        """
        self.envs[0].switch_task(task_class, target_position)

    def reset(self) -> Dict[str, np.ndarray]:
        descriptions, obs = self.task.reset()
        self.n_steps = 0
        del descriptions  # Not used.
        return self._extract_obs(obs)

    def normalize_action(self, action: np.ndarray):
        """
        Normalizes desired orientation or change in orientation of EE. Also normalizes change in position if
        control type is DELTA_EE. Only should be called if 
        action controls EE pose or change in pose. Actions have the following form:
        [x, y, z, qx, qy, qz, qw, gripper]
        """
        [ax, ay, az] = action[:3]
        x, y, z, qx, qy, qz, qw = self.task._robot.arm.get_tip().get_pose()
        cur_pos = np.array([x, y, z])
        cur_ori = np.array([qx, qy, qz, qw])

        # position
        d_pos = np.array([ax, ay, az])
        d_pos /= (np.linalg.norm(d_pos) * 100.0)

        # orientation
        d_quat = np.array([0, 0, 0, 1.0])
        # d_euler = action[3:6] / 10.0
        # drho, dphi, dtheta = d_euler
        # rot = R.from_euler("xyz", [drho, dphi, dtheta], degrees=True)
        # d_quat = rot.as_quat()

        # gripper_open = action[-1]
        gripper_open = 1.0

        if self.task._action_mode.arm in self.delta_ee_control_types:
            action = np.concatenate([d_pos, d_quat, [gripper_open]])

            # try:
            #     joint_positions = self.task._robot.arm.solve_ik(
            #         action[:3], quaternion=action[3:-1], relative_to=self.task._robot.arm.get_tip())
            #     print("target joint positions: %s" % np.array2string(np.array(joint_positions), precision=2))
            # except Exception as e:
            #     print("Failed to get target joint posiitions due to %s" % str(e))

        else:
            new_pos = cur_pos + d_pos
            new_quat = cur_ori
            action = np.concatenate([new_pos, new_quat, [1.0]])

        return action

    def manual_step(self, action):
        self.task._robot.arm.get_tip().set_pose(action[:-1])
        success, terminate = self.task._task.success()
        task_reward = self.task._task.reward()
        obs = self._extract_obs(self.task._scene.get_observation())
        return obs, task_reward, terminate

    def select_only_position(self, action: np.ndarray, action_range: float):

        # import ipdb; ipdb.set_trace()
        action = np.clip(action, -action_range, action_range)

        mask = np.zeros(action.shape)
        mask[:3] = 1

        action = action * mask
        action[6] = 1

        return action

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:

        self.n_steps += 1

        if self.task._action_mode.arm in self.ee_control_types:
            action = self.normalize_action(action)

        # try:
        #     obs, reward, terminate = self.manual_step(action)
        #     # self.task._scene.step()
        # except Exception as e:
        #     print(e)

        terminate = False
        try:
            obs, reward, success = self.task.step(action)
            obs = self._extract_obs(obs)
        except pyrep.errors.ConfigurationPathError as e:
            # print("Action %s failed due to %s" % (np.array2string(action, precision=3), e))
            obs = self._extract_obs(self.task._scene.get_observation())
            _, success = self.task._task.success()
            reward = self.task._task.reward()
            # scale reward by change in translation/rotation
        except rlbench.task_environment.InvalidActionError as e:
            # print("Action %s failed due to %s" % (np.array2string(action, precision=3), e))
            obs = self._extract_obs(self.task._scene.get_observation())
            _, success = self.task._task.success()
            if self.penalize_illegal:
                reward = -5
            else:
                reward = self.task._task.reward()

            if self.manual_terminate:
                terminate = True
                self.reset()

        if self.n_steps > self.epsiode_length:
            self.reset()
            terminate = True

        if success:
            print("Reached Goal!")

        return obs, reward, terminate or success, {'is_success': success}

    def close(self) -> None:
        self.env.shutdown()
