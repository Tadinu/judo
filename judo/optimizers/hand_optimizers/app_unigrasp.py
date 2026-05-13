from typing import Callable, Optional

# hydra
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Torch
import torch

TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(torch.device(TORCH_DEVICE))
torch.set_default_dtype(torch.float32)

from xvfbwrapper import Xvfb

# mjmanip
from mjmanip.robot.arm_hand import ArmHandDiffIK
from mjmanip.mj_utils import (mj_model_joints_qids, mj_model_actuators_id_list, mj_data_move_mocap, mj_scene_clear,
                              mj_scene_draw_spheres, \
                              mj_data_mocap_pose)
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig
from mjmanip.control.fabrics.fabrics_controller import FabricsController
from mjmanip.robot.panda_leap_fabrics import (ARM_XML_PATH as PANDA_LEAP_FABRICS_ARM_XML_PATH,
                                              HAND_XML_PATH as PANDA_LEAP_FABRICS_HAND_XML_PATH, \
                                              PandaLeapWithFabricsEnv, PandaLeapWithFabrics)

# judo
from judo import PACKAGE_ROOT, BackendType
from judo.app.mpc_app import MPCApp
from judo.tasks.panda_leap_pick import USE_LEAP_MJX

if USE_LEAP_MJX:
    # NOTE: LeapMjx hand is more robust than Leap, so use [panda_leap_mjx] for now!
    from mjmanip.robot.leap_mjx import LeapMjx
    from mjmanip.robot.panda_leap_mjx import PandaLeapMjxEnv, PandaLeapMjx, ARM_SCENE_XML_PATH, ARM_XML_PATH, \
        HAND_XML_PATH

    PANDA_LEAP = PandaLeapMjx
    PANDA_LEAP_ENV = PandaLeapMjxEnv
else:
    from mjmanip.robot.panda_leap import PandaLeapEnv, PandaLeap, ARM_SCENE_XML_PATH, ARM_XML_PATH, HAND_XML_PATH

    PANDA_LEAP = PandaLeap
    PANDA_LEAP_ENV = PandaLeapEnv
PANDA_LEAP.NINSTANCES = 1

RECORD_TIME = 300
OBJ_NAME = PANDA_LEAP.OBJECT_NAMES[0]

FABRICS_CONFIGS_DIR = f"{PACKAGE_ROOT}/configs/fabrics"

cs = ConfigStore.instance()
cs.store(name="panda_leap_mujoco", node=ArmHandPoseFabricConfig)

panda_leap_fabric_cfg_name = "panda_leap_mujoco"
panda_leap_fabric_cfg = None


@hydra.main(version_base=None, config_path=FABRICS_CONFIGS_DIR, config_name=panda_leap_fabric_cfg_name)
def fetch_fabric_cfg(cfg: DictConfig) -> None:
    global panda_leap_fabric_cfg
    panda_leap_fabric_cfg = OmegaConf.to_object(cfg)
    assert isinstance(panda_leap_fabric_cfg, ArmHandPoseFabricConfig)
    # print(OmegaConf.to_yaml(panda_leap_fabric_cfg))


class UniGraspApp(MPCApp):
    def __init__(self, task_name: str,
                 sim_backend_type: BackendType,
                 wp_kernel_set_joint_targets: Optional[Callable] = None,
                 fabric_cfg: Optional[ArmHandPoseFabricConfig] = None,
                 headless: bool = False) -> None:
        super().__init__(task_name, optimizer_name="UniGrasp", sim_backend_type=sim_backend_type,
                         robot_class=PANDA_LEAP,
                         wp_kernel_set_joint_targets=wp_kernel_set_joint_targets,
                         fabric_cfg=fabric_cfg,
                         kinematics_mode=False,
                         headless=headless)
        """Initialize the simulation node."""

    def config_fabrics(self):
        self.fabrics_env_class = PandaLeapWithFabricsEnv
        self.fabrics_robot_class = PandaLeapWithFabrics
        self.fabrics_arm_xml = PANDA_LEAP_FABRICS_ARM_XML_PATH
        self.fabrics_hand_xml = PANDA_LEAP_FABRICS_HAND_XML_PATH

    def plan(self) -> None:
        """Updates the control state internally."""
        if self.sim.paused:
            return
        self.plan_arm_hand()


def run_app(headless: bool) -> None:
    # NOTE: ["panda_leap_pick"] -> Key to [PandaLeapPickConfig], of which [mj_compose_spec()] loads XML -> mj_model
    app = UniGraspApp(task_name="panda_leap_pick", sim_backend_type=BackendType.MUJOCO,
                      fabric_cfg=panda_leap_fabric_cfg,
                      headless=headless)
    app.spin()


if __name__ == "__main__":
    fetch_fabric_cfg()
    app_headless = False
    if app_headless:
        with Xvfb(width=1920, height=1080) as xvfb:
            print(f"Using Xvfb display: {xvfb.new_display}")
            run_app(app_headless)
    else:
        run_app(app_headless)
