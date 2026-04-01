import time
from typing import Callable, Optional
from loop_rate_limiters import RateLimiter

import torch

TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(torch.device(TORCH_DEVICE))
torch.set_default_dtype(torch.float32)

import numpy as np
import warp as wp
import mujoco as mj
from mujoco import viewer
import roma
from xvfbwrapper import Xvfb

# mjmanip
from mjmanip.robot.arm_hand import ArmHandDiffIK
from mjmanip.utils import mj_get_joints_qids, mj_get_actuators_id_list, mj_move_mocap, mj_clear_scene, mj_draw_spheres

# judo
from judo import BackendType
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.nt_simulation import NTSimulation
from judo.app.utils import set_seed
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

# hand optimizer
from hand_optimizer import HandOptimizer, HandOptimizerParams, HandParams
from object_utils import ObjectData

RECORD_TIME = 300
OBJ_NAME = PANDA_LEAP.OBJECT_NAMES[0]
PANDA_LEAP.BASE_PLATFORM_NAME = "base_platform"


class UniGraspApp:
    def __init__(self, task_name: str,
                 sim_backend_type: BackendType,
                 wp_kernel_set_joint_targets: Optional[Callable] = None,
                 headless: bool = False) -> None:
        """Initialize the simulation node."""
        self.task_name = task_name
        self.step_cnt = 0
        self.num_rollouts = 1
        self.qpos_home: Optional[np.ndarray] = PANDA_LEAP.ARM_HOME_QPOS + PANDA_LEAP.HAND_HOME_QPOS + \
                                               PANDA_LEAP.OBJECT_INIT_POSES[OBJ_NAME].tolist()

        # 1- Sim
        kinematics_mode = False
        match sim_backend_type:
            case BackendType.MUJOCO | BackendType.MUJOCO_WARP:
                self.sim = MJSimulation(init_task=task_name,
                                        num_rollout_worlds=self.num_rollouts,
                                        kinematics_mode=kinematics_mode,
                                        headless=headless,
                                        record_video=headless)
            case BackendType.NEWTON:
                self.sim = NTSimulation(init_task=task_name,
                                        num_substeps=8,  # MPC
                                        num_rollout_worlds=self.num_rollouts,
                                        wp_kernel_set_joint_targets=wp_kernel_set_joint_targets,
                                        kinematics_mode=kinematics_mode)
        self.mj_model = self.sim.task.mj_model
        self.mj_data = self.sim.task.mj_data
        self.mj_robot_ctrl = self.mj_data.qpos if self.sim.kinematics_mode else self.mj_data.ctrl
        full_hand_base_name = PANDA_LEAP.hand_item_full_name(PANDA_LEAP.HAND_BASE_NAME)
        self.hand_base = self.mj_data.body(full_hand_base_name)
        self.obj = self.mj_data.body(OBJ_NAME)
        self.base_platform = self.mj_data.body(PANDA_LEAP.BASE_PLATFORM_NAME)
        mj.mj_forward(self.mj_model, self.mj_data)

        # 2- Controllers
        # Arm controller
        self.arm_qpos_ids = mj_get_joints_qids(self.mj_model, PANDA_LEAP.ARM_JOINTS_NAMES, is_qpos=True)
        self.arm_ctrl_ids = mj_get_actuators_id_list(self.mj_model, PANDA_LEAP.ARM_ACTS_NAMES)
        self.hand_qpos_ids = mj_get_joints_qids(self.mj_model,
                                                PANDA_LEAP.hand_items_full_names(PANDA_LEAP.HAND_JOINTS_NAMES),
                                                is_qpos=True)
        self.hand_ctrl_ids = mj_get_actuators_id_list(self.mj_model,
                                                      PANDA_LEAP.hand_items_full_names(PANDA_LEAP.HAND_ACTS_NAMES))
        self.diff_ik = ArmHandDiffIK(self.mj_model, self.mj_data, PANDA_LEAP, self.qpos_home,
                                     ee_name=full_hand_base_name, ee_obj_type='body')
        self.diff_ik.DT = self.mj_model.opt.timestep
        self.diff_ik.init()

        # Hand controller
        set_seed(0)
        self.opt_params = HandOptimizerParams(n_batches=1, distance_lower=0.05, distance_upper=0.15,
                                              jitter_strength=0.1,
                                              joint_limit_lower=-np.pi / 6,
                                              joint_limit_upper=np.pi / 6)

        self.object_data = ObjectData.get_mj_object_data(self.mj_model, self.mj_data,
                                                         body_names=[self.obj.name], device=TORCH_DEVICE)
        self.base_plate_data = ObjectData.get_mj_object_data(self.mj_model, self.mj_data,
                                                             body_names=[self.base_platform.name], device=TORCH_DEVICE)

        self.hand_opt = HandOptimizer(hand_params=HandParams.get(hand_model_name='leap_hand',
                                                                 xml_path=HAND_XML_PATH,
                                                                 joint_angles=np.array(
                                                                     self.mj_data.qpos[self.hand_qpos_ids],
                                                                     dtype=np.float32),
                                                                 hand_pos=self.hand_base.xpos.copy(),
                                                                 hand_quat=self.hand_base.xquat.copy()),
                                      object_data=self.object_data,
                                      obstacle_data=self.base_plate_data,
                                      opt_params=self.opt_params,
                                      device=TORCH_DEVICE)

    @property
    def is_mujoco(self):
        return isinstance(self.sim, MJSimulation)

    @property
    def is_newton(self):
        return isinstance(self.sim, NTSimulation)

    def spin(self) -> None:
        """Spin logic for the simulation node."""
        if self.is_mujoco:
            self.mj_spin()
        elif self.is_newton:
            self.nt_spin()

    def mj_spin(self):
        main_model = self.sim.task.mj_sim_model
        main_data = self.sim.task.mj_data
        rate = RateLimiter(frequency=1 / main_model.opt.timestep, warn=False)
        with mj.viewer.launch_passive(model=main_model, data=main_data, show_left_ui=False,
                                      show_right_ui=False) as mj_viewer:
            self.sim.mj_viewer = mj_viewer
            mj.mjv_defaultFreeCamera(main_model, mj_viewer.cam)
            while mj_viewer.is_running():
                mj.mj_camlight(main_model, main_data)

                # Plan
                self.plan()

                # Step
                self.sim.step()
                if self.sim.headless:
                    self.step_cnt += 1
                    if self.step_cnt >= RECORD_TIME:
                        break
                rate.sleep()

            # Close sim
            self.sim.close()

    def nt_spin(self):
        nt_viewer = self.sim.sim_backend.viewer
        while nt_viewer.is_running():
            with wp.ScopedTimer("step", active=False):
                start_time = time.time()
                self.plan()
                self.sim.step()

            with wp.ScopedTimer("render", active=False):
                self.sim.sim_backend.render()

            # Force simulation node to run at fixed rate specified by simulation timestep (specified in the model).
            dt_des = self.sim.timestep
            dt_elapsed = time.time() - start_time
            if dt_elapsed < dt_des:
                time.sleep(dt_des - dt_elapsed)
            else:
                print(f"Newton Sim step {dt_elapsed:.3f} longer than desired step {dt_des:.3f}!")

        # Close viewer
        nt_viewer.close()

    def plan(self) -> None:
        """Updates the controls state internally."""
        if self.sim.paused:
            return

        # start = time.perf_counter()
        # Hand plan
        hand_batches_num = self.hand_opt.n_batches
        assert hand_batches_num == 1
        cur_hand_pos = self.hand_base.xpos
        cur_hand_quat = self.hand_base.xquat
        cur_wrist_rot = torch.tensor(cur_hand_quat).repeat(hand_batches_num, 1) if self.hand_opt.use_quat \
            else (roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(torch.tensor(cur_hand_quat)))
                  .repeat(hand_batches_num, 1, 1))
        next_grasp = self.hand_opt.step_optimize(cur_wrist_pos=np.tile(cur_hand_pos, (hand_batches_num, 1)),
                                                 cur_wrist_rot=cur_wrist_rot,
                                                 cur_obj_mesh_poses=[np.concatenate([self.obj.xpos, self.obj.xquat])],
                                                 cur_obst_mesh_poses=[
                                                     np.concatenate(
                                                         [self.base_platform.xpos, self.base_platform.xquat])])
        self.mj_robot_ctrl[self.hand_ctrl_ids] = next_grasp.joint_angles
        mj_move_mocap(self.mj_model, self.mj_data, PANDA_LEAP.EE_TARGET_MOCAP_NAME,
                      pos=next_grasp.wrist_pos, quat=next_grasp.wrist_quat)

        # Arm plan
        next_grasp_pose = next_grasp.wrist_pose
        q = self.diff_ik.plan(target_ee_pose=next_grasp_pose, use_solver=True)
        if q is None:
            q = self.diff_ik.plan(target_ee_pose=next_grasp_pose, use_solver=False)
        self.mj_robot_ctrl[self.arm_ctrl_ids] = q[self.arm_qpos_ids]

        # Visualize
        visualize_grasp = False
        if visualize_grasp:
            self.hand_opt.visualize_grasp(next_grasp, self.object_data.meshes)

        # Traces
        # Hand pcl
        self.visualize_hand_pcl()

        # Obj pcl
        self.visualize_obj_pcl()

        # end = time.perf_counter()

    def visualize_hand_pcl(self):
        mj_draw_spheres(self.sim.mj_viewer.user_scn,
                        positions=self.hand_opt.hand_verts.tolist(),
                        sizes=len(self.hand_opt.hand_verts) * [[0.005]],
                        rgbas=len(self.hand_opt.hand_verts) * [[1, 1, 0, 1]])

    def visualize_obj_pcl(self):
        obj_points = self.hand_opt.object_data.all_points.detach().cpu().numpy().tolist()
        mj_draw_spheres(self.sim.mj_viewer.user_scn,
                        positions=obj_points,
                        sizes=len(obj_points) * [[0.005]],
                        rgbas=len(obj_points) * [[0, 1, 0, 1]])

    def toggle_paused_status(self) -> None:
        """Event handler for processing pause status updates."""
        self.sim.paused = not self.sim.paused


def run_app(headless: bool) -> None:
    app = UniGraspApp(task_name="panda_leap_pick", sim_backend_type=BackendType.MUJOCO, headless=headless)
    app.spin()


if __name__ == "__main__":
    headless = False
    if headless:
        with Xvfb(width=1920, height=1080) as xvfb:
            print(f"Using Xvfb display: {xvfb.new_display}")
            run_app(headless)
    else:
        run_app(headless)
