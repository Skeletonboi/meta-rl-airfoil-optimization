from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition
from rlbench.tasks import ReachTarget
import numpy as np

import ipdb


class ReachTargetCustom(ReachTarget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_position = None

    def reward(self) -> float:
        dist = self.target.get_position(self.robot.arm.get_tip())
        dist_val = np.linalg.norm(dist) / 10.0
#         if dist_val < 0.1:
#             return 1
        return -dist_val

    def init_episode(self, index: int) -> List[str]:
        super().init_episode(index)

        if self.target_position is not None:
            self.target.set_position(self.target_position)

    def get_name(self):
        return "reach_target"
