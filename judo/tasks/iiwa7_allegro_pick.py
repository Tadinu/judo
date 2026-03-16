from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

# mujoco
import mujoco as mj

# mjmanip
from mjmanip.robot.iiwa7_allegro_fabrics import IIWA7AllegroFabricsEnv, IIWA7AllegroBiotac, ARM_SCENE_XML_PATH, \
    ARM_XML_PATH, HAND_XML_PATH
from mjmanip.utils import mj_body_free_joint_name

# judo
from judo import BackendType
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.utils.math_utils import np_quat_diff_so3
from judo.utils.mujoco import mj_get_qpos_ids, mj_get_mocap_id

if TYPE_CHECKING:
    from judo.simulation.base import Simulation

OBJ_NAME = IIWA7AllegroBiotac.OBJECT_NAMES[0]


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class IIWA7AllegroPickConfig(TaskConfig):
    """Reward configuration IIWA7-ALLEGRO obj picking task."""

    task_name: str = "iiwa7_allegro_pick"
    sim_backend: str = BackendType.MUJOCO.name

    xml_path = None
    sim_xml_path = None
    qpos_home: Optional[np.ndarray] = field(default_factory=lambda:
    IIWA7AllegroBiotac.ARM_HOME_QPOS +
    IIWA7AllegroBiotac.HAND_HOME_QPOS +
    IIWA7AllegroBiotac.OBJECT_INIT_POSES[OBJ_NAME].tolist())  # fmt: skip
    reset_command: Optional[np.ndarray] = field(default_factory=lambda: np.array(
        IIWA7AllegroBiotac.ARM_HOME_QPOS +
        IIWA7AllegroBiotac.HAND_HOME_QPOS))  # fmt: skip
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.03, 0.1]))
    goal_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    w_pos: float = 0.1
    w_rot: float = 50

    def __post_init__(self):
        self.joint_names = []


class IIWA7AllegroPick(Task[IIWA7AllegroPickConfig]):
    """Defines the ALLEGRO cube rotation task."""

    config_t: type[IIWA7AllegroPickConfig] = IIWA7AllegroPickConfig

    def __init__(self, sim: Optional[Simulation] = None, num_rollout_worlds: int = 1) -> None:
        """Initializes the ALLEGRO cube rotation task."""

        super().__init__(sim, num_rollout_worlds=num_rollout_worlds)
        self.goal_pos = self.config.goal_pos
        self.goal_quat = self.config.goal_quat
        self.qpos_home = self.config.qpos_home
        self.obj_id: int = -1
        self.obj_qpos_ids: list[int]
        self.target_mocap_id: int = -1

    def init_ids(self):
        super().init_ids()
        self.obj_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_BODY, OBJ_NAME)
        self.obj_qpos_ids = mj_get_qpos_ids(self.mj_model, [mj_body_free_joint_name(OBJ_NAME)])
        self.target_mocap_id = mj_get_mocap_id(self.mj_model, IIWA7AllegroBiotac.goal_name(OBJ_NAME))

        # distance sensors
        # NOTE: For rollout result analysis, these are only valid IF MuJoCo-C Rollout backend supports mocap_pos/quat
        # for the `initial_state`
        # distance sensors
        self.obj_pos_distance_to_grasp_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_distance_to_grasp")
        self.obj_quat_distance_sensor_idx = self.get_sensor_start_index(f"{OBJ_NAME}_orientation_distance_to_goal")

    def mj_compose_spec(self) -> Optional[mj.MjSpec]:
        env = IIWA7AllegroFabricsEnv(world_scene_xml=ARM_SCENE_XML_PATH,
                                     arm_hand_class=IIWA7AllegroBiotac,
                                     arm_xml=ARM_XML_PATH,
                                     hand_xml=HAND_XML_PATH,
                                     fabric_cfg=None)
        return env.construct_main_spec(meshdir=env.meshdir, texturedir=env.texturedir)

    def reward(self,
               states: np.ndarray,
               sensors: np.ndarray,
               controls: np.ndarray,
               system_metadata: Optional[dict[str, Any]] = None) -> np.ndarray:
        """Implements the ALLEGRO cube rotation tracking task reward."""
        if system_metadata is None:
            system_metadata = {}
        goal_quat = system_metadata.get("goal_quat", self.config.goal_quat)

        # weights
        w_pos = self.config.w_pos
        w_rot = self.config.w_rot

        # "standard" tracking task
        qo_pos_traj = states[..., :3]
        qo_quat_traj = states[..., 3:7]
        qo_pos_diff = qo_pos_traj - self.config.goal_pos
        qo_quat_diff = np_quat_diff_so3(qo_quat_traj, goal_quat)

        pos_cost = w_pos * 0.5 * np.square(qo_pos_diff).sum(-1).mean(-1)
        rot_cost = w_rot * 0.5 * np.square(qo_quat_diff).sum(-1).mean(-1)
        rewards = -(pos_cost + rot_cost)
        return rewards

    def pre_sim_step(self) -> None:
        self._update_goal()

    def _update_goal(self) -> None:
        self.goal_pos = self.mj_data.mocap_pos[self.target_mocap_id]
        self.goal_quat = self.mj_data.mocap_quat[self.target_mocap_id]

    def reset(self) -> None:
        """Resets the model to a default state with random goal."""
        if self.mj_model:
            """Resets the model to a default state with random goal."""
            self.mj_data.qpos[:] = self.qpos_home
            self.mj_data.qvel[:] = 0.0
            self.mj_data.ctrl[:] = self.config.reset_command
            self._update_goal()
            mj.mj_forward(self.mj_model, self.mj_data)
        else:
            pass

    def get_sim_metadata(self) -> dict[str, Any]:
        """Returns the simulation's goal quat."""
        return {"goal_quat": self.goal_quat}
