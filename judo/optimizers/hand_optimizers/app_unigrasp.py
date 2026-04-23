from typing import Callable, Optional
import torch

TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(torch.device(TORCH_DEVICE))
torch.set_default_dtype(torch.float32)

from xvfbwrapper import Xvfb

# mjmanip
from mjmanip.robot.arm_hand import ArmHandDiffIK
from mjmanip.utils import mj_get_joints_qids, mj_get_actuators_id_list, mj_move_mocap, mj_clear_scene, mj_draw_spheres

# judo
from judo import BackendType
from judo.app.mpc_app import MPCApp
from judo.tasks.panda_leap_pick import USE_LEAP_MJX

if USE_LEAP_MJX:
    # NOTE: LeapMjx hand is more robust than Leap, so use [panda_leap_mjx] for now!
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
PANDA_LEAP.BASE_PLATFORM_NAME = "base_platform"


class UniGraspApp(MPCApp):
    def __init__(self, task_name: str,
                 sim_backend_type: BackendType,
                 wp_kernel_set_joint_targets: Optional[Callable] = None,
                 headless: bool = False) -> None:
        super().__init__(task_name, optimizer_name="UniGrasp", sim_backend_type=sim_backend_type,
                         robot_class=PANDA_LEAP,
                         wp_kernel_set_joint_targets=wp_kernel_set_joint_targets, headless=headless,
                         kinematics_mode=False)
        """Initialize the simulation node."""

    def plan(self) -> None:
        """Updates the controls state internally."""
        if self.sim.paused:
            return
        super().plan_arm_hand()


def run_app(headless: bool) -> None:
    app = UniGraspApp(task_name="panda_leap_pick", sim_backend_type=BackendType.MUJOCO, headless=headless)
    app.spin()


if __name__ == "__main__":
    app_headless = False
    if app_headless:
        with Xvfb(width=1920, height=1080) as xvfb:
            print(f"Using Xvfb display: {xvfb.new_display}")
            run_app(app_headless)
    else:
        run_app(app_headless)
