import os
import time
import random
from typing import Callable, Optional
from loop_rate_limiters import RateLimiter

import torch

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float32)
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import warp as wp
import mujoco as mj
from mujoco import viewer
import roma
from xvfbwrapper import Xvfb

# mjmanip
from mjmanip.robot.arm_hand import ArmHandDiffIK
# NOTE: LeapMjx hand is more robust than Leap, so use [panda_leap_mjx] for now!
# from mjmanip.robot.panda_leap import PandaLeapEnv, PandaLeap, ARM_SCENE_XML_PATH, ARM_XML_PATH, HAND_XML_PATH
from mjmanip.robot.panda_leap_mjx import PandaLeapMjxEnv, PandaLeapMjx, ARM_SCENE_XML_PATH, ARM_XML_PATH, HAND_XML_PATH
from mjmanip.utils import mj_get_joints_qids, mj_get_actuators_id_list, mj_move_mocap, mj_clear_scene, mj_draw_spheres

if PandaLeapMjx:
    PandaLeapMjx.NINSTANCES = 1
    PandaLeap = PandaLeapMjx
if PandaLeapMjxEnv:
    PandaLeapEnv = PandaLeapMjxEnv

# judo
from judo import BackendType, PACKAGE_ROOT
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.nt_simulation import NTSimulation

# hand optimizer
from hand_optimizer import HandOptimizer, HandOptimizerParams, HandParams
from object_utils import ObjectData
from rot6d import compute_rotation_ortho6d_from_matrix

RECORD_TIME = 300
OBJ_NAME = PandaLeap.OBJECT_NAMES[0]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class UniGraspApp:
    def __init__(self, task_name: str,
                 sim_backend_type: BackendType,
                 kernel_set_joint_targets: Optional[Callable] = None,
                 headless: bool = False) -> None:
        """Initialize the simulation node."""
        self.task_name = task_name
        self.step_cnt = 0
        self.num_rollouts = 1
        self.qpos_home: Optional[np.ndarray] = PandaLeap.ARM_HOME_QPOS + PandaLeap.HAND_HOME_QPOS + \
                                               PandaLeap.OBJECT_INIT_POSES[OBJ_NAME].tolist()

        # 1- Sim
        match sim_backend_type:
            case BackendType.MUJOCO | BackendType.MUJOCO_WARP:
                self.sim = MJSimulation(init_task=task_name,
                                        num_rollout_worlds=self.num_rollouts,
                                        headless=headless,
                                        record_video=headless)
            case BackendType.NEWTON:
                self.sim = NTSimulation(init_task=task_name,
                                        num_substeps=8,  # MPC
                                        num_rollout_worlds=self.num_rollouts,
                                        kernel_set_joint_targets=kernel_set_joint_targets)
        self.mj_model = self.sim.task.mj_model
        self.mj_data = self.sim.task.mj_data
        full_hand_base_name = PandaLeap.hand_item_full_name(PandaLeap.HAND_BASE_NAME)
        self.hand_base = self.mj_data.body(full_hand_base_name)
        self.obj = self.mj_data.body(OBJ_NAME)
        mj.mj_forward(self.mj_model, self.mj_data)

        # 2- Controllers
        # Arm controller
        self.arm_qpos_ids = mj_get_joints_qids(self.mj_model, PandaLeap.ARM_JOINTS_NAMES, is_qpos=True)
        self.arm_ctrl_ids = mj_get_actuators_id_list(self.mj_model, PandaLeap.ARM_ACTS_NAMES)
        self.hand_qpos_ids = mj_get_joints_qids(self.mj_model,
                                                PandaLeap.hand_items_full_names(PandaLeap.HAND_JOINTS_NAMES),
                                                is_qpos=True)
        self.hand_ctrl_ids = mj_get_actuators_id_list(self.mj_model,
                                                      PandaLeap.hand_items_full_names(PandaLeap.HAND_ACTS_NAMES))
        self.diff_ik = ArmHandDiffIK(self.mj_model, self.mj_data, PandaLeap, self.qpos_home,
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
                                                         body_names=['obj'], device=torch_device)

        self.hand_opt = HandOptimizer(device=torch_device,
                                      object_data=self.object_data,
                                      hand_params=HandParams.get(hand_model_name='leap_hand',
                                                                 xml_path=HAND_XML_PATH,
                                                                 joint_angles=np.array(
                                                                     self.mj_data.qpos[self.hand_qpos_ids],
                                                                     dtype=np.float32),
                                                                 hand_pos=self.hand_base.xpos,
                                                                 hand_quat=self.hand_base.xquat),
                                      opt_params=self.opt_params)

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
                                                 cur_mesh_poses=[np.concatenate([self.obj.xpos, self.obj.xquat])])
        self.mj_data.ctrl[self.hand_ctrl_ids] = next_grasp.joint_angles
        mj_move_mocap(self.mj_model, self.mj_data, PandaLeap.EE_TARGET_MOCAP_NAME,
                      pos=next_grasp.wrist_pos, quat=next_grasp.wrist_quat)

        # Arm plan
        next_grasp_pose = next_grasp.wrist_pose
        q = self.diff_ik.plan(target_ee_pose=next_grasp_pose, use_solver=True)
        if q is None:
            q = self.diff_ik.plan(target_ee_pose=next_grasp_pose, use_solver=False)
        self.mj_data.ctrl[self.arm_ctrl_ids] = q[self.arm_qpos_ids]

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
