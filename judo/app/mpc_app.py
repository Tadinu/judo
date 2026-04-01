import time
import random
from typing import Callable, Optional
from threading import Lock
from omegaconf import DictConfig
from loop_rate_limiters import RateLimiter

import numpy as np
import warp as wp

import torch

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float32)
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import roma

import mujoco as mj
from mujoco import viewer

# judo
from judo import BackendType
from judo.config import get_override_config
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.nt_simulation import NTSimulation
from judo.controller import MPController, make_controller
from judo.utils.fabrics_utils import FabricsAgent, FABRICS_MPC_TYPE
from judo.app.structs import SplineData
from judo.app.utils import get_class_from_string

# mjmanip
from mjmanip.robot.arm_hand import ArmHand, ArmHandDiffIK
from mjmanip.utils import mj_get_joints_qids, mj_get_actuators_id_list, mj_move_mocap, mj_clear_scene, mj_draw_spheres
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig

# hand optimizer
from judo.optimizers.hand_optimizers.hand_optimizer import HandOptimizer, HandOptimizerParams, HandParams
from judo.optimizers.hand_optimizers.object_utils import ObjectData
from judo.app.utils import set_seed

RECORD_TIME = 300


class MPCApp:
    def __init__(self, task_name: str,
                 optimizer_name: str,
                 sim_backend_type: BackendType,
                 robot_class: Optional[ArmHand] = None,
                 wp_kernel_set_joint_targets: Optional[Callable] = None,
                 task_registration_cfg: Optional[DictConfig] = None,
                 optimizer_registration_cfg: Optional[DictConfig] = None,
                 fabric_cfg: Optional[ArmHandPoseFabricConfig] = None,
                 kinematics_mode: bool = False,
                 headless: bool = False) -> None:
        """Initialize the simulation node."""
        self.task_name = task_name
        self.step_cnt = 0
        self.robot_class: ArmHand = robot_class

        # 1- Sim
        optimizer_config_cls = get_class_from_string(optimizer_registration_cfg[optimizer_name].config)
        num_rollouts = get_override_config(optimizer_config_cls, task_name)["num_rollouts"]
        match sim_backend_type:
            case BackendType.MUJOCO | BackendType.MUJOCO_WARP:
                self.sim = MJSimulation(init_task=task_name,
                                        num_rollout_worlds=num_rollouts,
                                        task_registration_cfg=task_registration_cfg,
                                        headless=headless,
                                        kinematics_mode=kinematics_mode,
                                        record_video=headless)
            case BackendType.NEWTON:
                self.sim = NTSimulation(init_task=task_name,
                                        num_substeps=8,  # MPC
                                        num_rollout_worlds=num_rollouts,
                                        wp_kernel_set_joint_targets=wp_kernel_set_joint_targets,
                                        kinematics_mode=kinematics_mode,
                                        task_registration_cfg=task_registration_cfg)

        # 2- Fabrics computation agent
        if fabric_cfg and FABRICS_MPC_TYPE:
            self.sim.fabrics_agent = FabricsAgent(self.sim.task.mj_sim_model, self.sim.task.mj_data,
                                                  fabric_cfg, num_rollout_worlds=1, num_fabrics_steps=1)
            self.sim.task.fabrics_agent = FabricsAgent(self.sim.task.mj_model, self.sim.task.mj_data,
                                                       fabric_cfg, num_rollout_worlds=num_rollouts,
                                                       num_fabrics_steps=100)
            self.sim.task.map_controls = self.sim.task.fabrics_agent.hand_pca_to_q \
                if FabricsAgent.USE_PCA_HAND_GRASP else self.sim.task.fabrics_agent.fabrics_plan
            print("FABRICS SUBSTEPS: Task Rollout", self.sim.task.fabrics_agent.num_fabrics_steps,
                  "Sim", self.sim.fabrics_agent.num_fabrics_steps)

        # 3- Controller
        # NOTE: Controller uses task's nu to initiate its mpc-algo/optimizer so must be after fabrics_agent,
        # which decides task's nu
        # Whether the controller runs alongside the sim
        self.synchronous_controller = True
        if self.synchronous_controller:
            # MPC
            self.sim.mpcontroller = self.mpcontroller = make_controller(
                sim=self.sim,
                init_task=self.sim.task,
                init_optimizer=optimizer_name,
                task_registration_cfg=task_registration_cfg,
                optimizer_registration_cfg=optimizer_registration_cfg,
                rollout_backend=sim_backend_type
            )
            print("CONTROLLER NUM_TIMESTEPS", self.mpcontroller.num_timesteps)
            self.fetch_nominal_control_spline()
            self.write_state_to_controller()
            # Only for task update if from another thread
            self.lock = Lock()

            # MUJOCO-Specific controllers
            if self.is_mujoco and robot_class:
                OBJ_NAME = robot_class.OBJECT_NAMES[0]
                # Arm controller
                self.mj_model = self.sim.task.mj_sim_model
                self.mj_data = self.sim.task.mj_data
                self.qpos_home: Optional[np.ndarray] = robot_class.ARM_HOME_QPOS + robot_class.HAND_HOME_QPOS + \
                                                       robot_class.OBJECT_INIT_POSES[OBJ_NAME].tolist()
                self.arm_qpos_ids = mj_get_joints_qids(self.mj_model, robot_class.ARM_JOINTS_NAMES, is_qpos=True)
                self.arm_ctrl_ids = mj_get_actuators_id_list(self.mj_model, robot_class.ARM_ACTS_NAMES)
                self.hand_qpos_ids = mj_get_joints_qids(self.mj_model,
                                                        robot_class.hand_items_full_names(
                                                            robot_class.HAND_JOINTS_NAMES),
                                                        is_qpos=True)
                self.hand_ctrl_ids = mj_get_actuators_id_list(self.mj_model,
                                                              robot_class.hand_items_full_names(
                                                                  robot_class.HAND_ACTS_NAMES))
                self.diff_ik = ArmHandDiffIK(self.mj_model, self.mj_data, robot_class, self.qpos_home,
                                             ee_name=robot_class.hand_item_full_name(robot_class.HAND_BASE_NAME),
                                             ee_obj_type='body')
                self.diff_ik.DT = self.mj_model.opt.timestep
                self.diff_ik.init()

                # Hand controller
                set_seed(0)
                self.opt_params = HandOptimizerParams(n_batches=1, distance_lower=0.05, distance_upper=0.15,
                                                      jitter_strength=0.1,
                                                      joint_limit_lower=-np.pi / 6,
                                                      joint_limit_upper=np.pi / 6)

                self.obj = self.mj_data.body(OBJ_NAME)
                self.object_data = ObjectData.get_mj_object_data(self.mj_model, self.mj_data,
                                                                 body_names=[self.obj.name], device=TORCH_DEVICE)

                self.base_platform = self.mj_data.body(robot_class.BASE_PLATFORM_NAME)
                self.base_plate_data = ObjectData.get_mj_object_data(self.mj_model, self.mj_data,
                                                                     body_names=[self.base_platform.name],
                                                                     device=TORCH_DEVICE)

                self.hand_base = self.mj_data.body(
                    self.robot_class.hand_item_full_name(self.robot_class.HAND_BASE_NAME))
                self.hand_opt = HandOptimizer(
                    hand_params=HandParams.get(hand_model_name=self.robot_class.HAND_MODEL_NAME,
                                               xml_path=self.robot_class.HAND_XML_PATH,
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
        num_steps = int(self.sim.task.fabrics_agent.num_fabrics_steps / 3) if self.sim.task.fabrics_agent else 1
        rate = RateLimiter(frequency=1 / main_model.opt.timestep, warn=False)
        with mj.viewer.launch_passive(model=main_model, data=main_data, show_left_ui=False,
                                      show_right_ui=False) as mj_viewer:
            self.sim.mj_viewer = mj_viewer
            mj.mjv_defaultFreeCamera(main_model, mj_viewer.cam)
            while mj_viewer.is_running():
                mj.mj_camlight(main_model, main_data)

                # Plan
                if self.synchronous_controller and (self.step_cnt % num_steps == 0 if num_steps > 1 else True):
                    self.write_state_to_controller()
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
                if self.synchronous_controller:
                    self.write_state_to_controller()
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

    def controller_spin(self):
        """Spin logic for the controller node."""
        assert not self.synchronous_controller
        while True:
            start_time = time.time()
            # TODO:Fetch state data from sim
            self.write_state_to_controller()
            self.plan()

            # Force controller to run at fixed rate specified by control_freq.
            sleep_dt = 1 / self.mpcontroller.controller_cfg.control_freq - (time.time() - start_time)
            time.sleep(max(0, sleep_dt))

    def plan(self) -> None:
        """Updates the controls state internally."""
        if self.sim.paused:
            return

        # start = time.perf_counter()
        # Rollout controller here-in!
        if self.sim.task.should_stop_mpc():
            self.plan_arm_hand()
        else:
            self.mpcontroller.update_action()
            if self.is_mujoco:
                mj_move_mocap(self.mj_model, self.mj_data, self.robot_class.EE_TARGET_MOCAP_NAME,
                              pos=self.hand_base.xpos, quat=self.hand_base.xquat)
                if self.hand_opt:
                    self.hand_opt.update_opt_wrist_pose(wrist_pos=torch.from_numpy(self.hand_base.xpos).unsqueeze(0)
                                                        .float().to(TORCH_DEVICE),
                                                        wrist_rot=torch.from_numpy(self.hand_base.xquat)
                                                        .unsqueeze(0).float().to(TORCH_DEVICE))
        # end = time.perf_counter()

        # print("plan_time", end - start)
        self.fetch_nominal_control_spline()
        # self.sim.nominal_action = self.controller.action(self.sim.sim_backend.sim_time)
        # print("best action", self.sim.optimal_action)

    def plan_arm_hand(self):
        # Hand plan
        hand_batches_num = self.hand_opt.n_batches
        assert hand_batches_num == 1
        cur_hand_pos = self.hand_base.xpos.copy()
        cur_hand_quat = self.hand_base.xquat.copy()
        cur_wrist_rot = torch.tensor(cur_hand_quat).repeat(hand_batches_num, 1) if self.hand_opt.use_quat \
            else (roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(torch.tensor(cur_hand_quat)))
                  .repeat(hand_batches_num, 1, 1))
        next_grasp = self.hand_opt.step_optimize(cur_wrist_pos=np.tile(cur_hand_pos, (hand_batches_num, 1)),
                                                 cur_wrist_rot=cur_wrist_rot,
                                                 cur_obj_mesh_poses=[np.concatenate([self.obj.xpos, self.obj.xquat])],
                                                 cur_obst_mesh_poses=[
                                                     np.concatenate(
                                                         [self.base_platform.xpos, self.base_platform.xquat])])
        self.mj_data.ctrl[self.hand_ctrl_ids] = next_grasp.joint_angles
        mj_move_mocap(self.mj_model, self.mj_data, self.robot_class.EE_TARGET_MOCAP_NAME,
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

    def update_control(self, nominal_spline_data: SplineData) -> None:
        """Event handler for processing controls received from controller node."""
        nominal_control = nominal_spline_data.spline()
        self.sim.update_nominal_control_spline(nominal_control)

    def update_task(self, task_name: str) -> None:
        """Updates the task type."""
        task_entry = self.mpcontroller.available_tasks.get(task_name)
        if task_entry is not None:
            self.task_name = task_name
            task_cls, _ = task_entry
            with self.lock:
                task = task_cls(self.sim)
                optimizer = self.mpcontroller.optimizer_cls(self.mpcontroller.optimizer_config_cls(), task.nu)
                self.mpcontroller = MPController(
                    controller_config=self.mpcontroller.controller_cfg,
                    task=task,
                    optimizer=optimizer,
                )
                self.fetch_nominal_control_spline()
        else:
            raise ValueError(f"Task {task_name} not found in task registry.")

    def update_optimizer(self, optimizer_name: str) -> None:
        """Updates the optimizer type."""
        optimizer_entry = self.mpcontroller.available_optimizers.get(optimizer_name)
        if optimizer_entry is not None:
            optimizer_cls, optimizer_config_cls = optimizer_entry
            optimizer_config = optimizer_config_cls()
            optimizer = optimizer_cls(optimizer_config, self.mpcontroller.task.nu)
            with self.lock:
                self.mpcontroller.optimizer = optimizer
        else:
            raise ValueError(f"Optimizer {optimizer_name} not found in optimizer registry.")

    def reset_task(self, event: dict) -> None:
        """Resets the task."""
        with self.lock:
            self.mpcontroller.reset()
            self.fetch_nominal_control_spline()

    def toggle_paused_status(self) -> None:
        """Event handler for processing pause status updates."""
        self.sim.paused = not self.sim.paused

    def write_state_to_controller(self) -> None:
        self.mpcontroller.update_state(self.sim.sim_state)

    def fetch_nominal_control_spline(self) -> None:
        """Util that publishes the current controller spline."""
        # Set sim's control spline from [self.controller]
        self.update_control(self.mpcontroller.nominal_spline_data)

        # Visualize traces
        if self.mpcontroller.traces is not None and len(self.mpcontroller.traces) > 0:
            pass

    def visualize_hand_pcl(self):
        if self.is_mujoco:
            mj_draw_spheres(self.sim.mj_viewer.user_scn,
                            positions=self.hand_opt.hand_verts.tolist(),
                            sizes=len(self.hand_opt.hand_verts) * [[0.005]],
                            rgbas=len(self.hand_opt.hand_verts) * [[1, 1, 0, 1]])

    def visualize_obj_pcl(self):
        if self.is_mujoco:
            obj_points = self.hand_opt.object_data.all_points.detach().cpu().numpy().tolist()
            mj_draw_spheres(self.sim.mj_viewer.user_scn,
                            positions=obj_points,
                            sizes=len(obj_points) * [[0.005]],
                            rgbas=len(obj_points) * [[0, 1, 0, 1]])
