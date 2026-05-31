import os
import time
from typing import Any, Callable, Optional, Union, Sequence
from threading import Lock
from omegaconf import DictConfig
from loop_rate_limiters import RateLimiter

import numpy as np
import numpy.typing as npt
import warp as wp

import torch

torch.set_default_device('cuda')
torch.set_default_dtype(torch.float32)
TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import roma

import mujoco as mj
from mujoco import viewer

# mjmanip
from mjmanip import OBJECT_MODELS_DIR as MJMANIP_OBJECT_MODELS_DIR
from mjmanip.robot.arm_hand import ArmHand, ArmHandDiffIK
from mjmanip.mj_utils import (mj_data_body_geoms, mj_model_joints_qids, mj_model_actuators_id_list, mj_data_move_mocap,
                              mj_scene_clear, mj_scene_draw_spheres, mj_data_mocap_pose, mj_data_geoms_global_poses,
                              mj_data_site_pose)
from mjmanip.pytorch3d_utils import p3d_multiply_poses, p3d_pose_inverse
from mjmanip.robot.arm_hand_fabrics import ArmHandWithFabricsEnv
from mjmanip.control.fabrics.fabrics_controller import FabricsController
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig

# judo
from judo import BackendType
from judo.config import get_override_config
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.nt_simulation import NTSimulation
from judo.controller import MPController, make_controller
from judo.utils.fabrics_utils import FabricsAgent, FABRICS_MPC_TYPE
from judo.app.structs import SplineData
from judo.app.utils import get_class_from_string

# hand optimizer
from judo.optimizers.hand_optimizers.hand_optimizer import HandOptimizer, HandOptimizerParams, HandParams, HandGrasp
from judo.optimizers.hand_optimizers.object_utils import ObjectData
from judo.app.utils import set_seed

RECORD_TIME = 300


class MPCApp:
    USE_DIFF_IK = False
    USE_CURRENT_HAND_BASE_CENTRIC_GRASP = True

    def __init__(self, task_name: str,
                 optimizer_name: str,
                 sim_backend_type: BackendType,
                 robot_class: Optional[ArmHand] = None,
                 wp_kernel_set_joint_targets: Optional[Callable] = None,
                 task_registration_cfg: Optional[DictConfig] = None,
                 optimizer_registration_cfg: Optional[DictConfig] = None,
                 fabric_cfg: Optional[ArmHandPoseFabricConfig] = None,
                 kinematics_mode: bool = False,
                 headless: bool = False,
                 device: str = TORCH_DEVICE) -> None:
        """Initialize the simulation node."""
        self.task_name = task_name
        self.step_cnt = 0
        self.robot_class: ArmHand = robot_class
        self.device = device

        # 1- Sim
        optimizer_config_cls = get_class_from_string(optimizer_registration_cfg[optimizer_name].config) \
            if optimizer_registration_cfg else None
        num_rollouts = get_override_config(optimizer_config_cls, task_name)["num_rollouts"] if optimizer_config_cls \
            else 1
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
        self.fabric_cfg = fabric_cfg
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

        # 3- Controllers
        # NOTE: Controller uses task's nu to initiate its mpc-algo/optimizer so must be after fabrics_agent,
        # which decides task's nu
        # Whether the controllers run alongside the sim
        self.synchronous_controlling = True
        if self.synchronous_controlling:
            # MPC
            if task_registration_cfg:
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
                self.mj_spec = self.sim.task.mj_spec
                self.mj_model = self.sim.task.mj_sim_model
                self.mj_data = self.sim.task.mj_data
                self.qpos_home: Optional[np.ndarray] = robot_class.ARM_HOME_QPOS + robot_class.HAND_HOME_QPOS + \
                                                       robot_class.OBJECT_INIT_POSES[OBJ_NAME].tolist()
                self.arm_qpos_ids = mj_model_joints_qids(self.mj_model, robot_class.ARM_JOINTS_NAMES, is_qpos=True)
                self.arm_dof_ids = mj_model_joints_qids(self.mj_model, robot_class.ARM_JOINTS_NAMES, is_qpos=False)
                self.arm_ctrl_ids = mj_model_actuators_id_list(self.mj_model, robot_class.ARM_ACTS_NAMES)
                self.hand_qpos_ids = mj_model_joints_qids(self.mj_model,
                                                          robot_class.hand_items_full_names(
                                                              robot_class.HAND_JOINTS_NAMES),
                                                          is_qpos=True)
                self.hand_dof_ids = mj_model_joints_qids(self.mj_model,
                                                         robot_class.hand_items_full_names(
                                                             robot_class.HAND_JOINTS_NAMES),
                                                         is_qpos=False)
                self.hand_ctrl_ids = mj_model_actuators_id_list(self.mj_model,
                                                                robot_class.hand_items_full_names(
                                                                    robot_class.HAND_ACTS_NAMES))
                self.robot_qpos_ids = np.concat([self.arm_qpos_ids, self.hand_qpos_ids])
                self.robot_dof_ids = np.concat([self.arm_dof_ids, self.hand_dof_ids])
                self.mj_robot_ctrl = self.mj_data.qpos if self.sim.kinematics_mode else self.mj_data.ctrl

                # Object
                OBJ_BODY_NAMES = [OBJ_NAME]
                OBJ_GEOM_NAMES = robot_class.OBJECT_GEOM_NAMES[OBJ_NAME]
                self.obj = self.mj_data.body(OBJ_NAME)
                self.obj_geoms = {geom_name: self.mj_data.geom(geom_name) for geom_name in OBJ_GEOM_NAMES}
                self.obj_visual_geom = self.obj_geoms[OBJ_GEOM_NAMES[0]]
                self.object_data = ObjectData.get_mj_object_data(OBJ_NAME, self.mj_model, self.mj_data, self.mj_spec,
                                                                 body_names=OBJ_BODY_NAMES,
                                                                 geom_names=OBJ_GEOM_NAMES,
                                                                 meshdir=str(os.path.join(MJMANIP_OBJECT_MODELS_DIR,
                                                                                          OBJ_NAME)),
                                                                 is_collision=True,
                                                                 merging_meshes=False,
                                                                 npoints_each_geom=10,
                                                                 device=self.device)

                # Obstacle
                OBJ_COLLISION_GEOM_NAMES = robot_class.OBJECT_COLLISION_GEOM_NAMES[OBJ_NAME]
                self.obstacle_geoms = {geom_name: self.mj_data.geom(geom_name) for geom_name in
                                       OBJ_COLLISION_GEOM_NAMES}
                self.obj_obstacle_data = ObjectData.get_mj_object_data(
                    f"{OBJ_NAME}_obstacles",
                    self.mj_model, self.mj_data, self.mj_spec,
                    body_names=OBJ_BODY_NAMES,
                    geom_names=OBJ_COLLISION_GEOM_NAMES,
                    meshdir=str(os.path.join(MJMANIP_OBJECT_MODELS_DIR, OBJ_NAME)),
                    is_collision=True,
                    merging_meshes=False,
                    npoints_each_geom=10,
                    device=self.device
                ) if OBJ_COLLISION_GEOM_NAMES else None

                # Env bodies (as one of obstacles)
                self.env_bodies = {env_body_name: self.mj_data.body(env_body_name)
                                   for env_body_name in robot_class.ENV_BODY_NAMES}
                # self.env_body_geoms = mj_data_body_geoms(self.mj_model, self.mj_data, env_body_name)
                self.env_bodies_data = ObjectData.get_mj_object_data("env_bodies",
                                                                     self.mj_model,
                                                                     self.mj_data, self.mj_spec,
                                                                     body_names=robot_class.ENV_BODY_NAMES,
                                                                     is_collision=True,
                                                                     npoints_each_geom=100,
                                                                     device=self.device)

                # IK controllers
                self.diff_ik: ArmHandDiffIK = None
                self.fabrics_env_class: type[ArmHandWithFabricsEnv] = None
                self.fabrics_env: ArmHandWithFabricsEnv = None
                self.fabrics_robot_class: type[ArmHand] = None
                self.fabrics_robot: ArmHand = None
                self.fabrics_arm_xml: str = ""
                self.fabrics_hand_xml: str = ""
                self.fabrics_controller: FabricsController = None
                self.init_ik_controllers()

                # Hand-base/Fingers pose optimizer
                set_seed(0)
                self.opt_params = HandOptimizerParams(nbatches=1, distance_lower=0.05, distance_upper=0.15,
                                                      jitter_strength=0.1,
                                                      joint_limit_lower=-np.pi / 6,
                                                      joint_limit_upper=np.pi / 6)
                self.hand_base = self.mj_data.body(robot_class.hand_item_full_name(self.robot_class.HAND_BASE_NAME))
                # Object-centric (VS hand-base centric) grasp initialization
                hand_params = HandParams(hand_model_name=robot_class.HAND_MODEL_NAME,
                                         xml_path=robot_class.HAND_XML_PATH,
                                         joint_angles=torch.from_numpy(np.array(self.mj_data.qpos[self.hand_qpos_ids],
                                                                                dtype=np.float32)).to(device).
                                         unsqueeze(0)) if True \
                    else HandParams.get(hand_model_name=robot_class.HAND_MODEL_NAME,
                                        xml_path=robot_class.HAND_XML_PATH,
                                        joint_angles=np.array(self.mj_data.qpos[self.hand_qpos_ids], dtype=np.float32),
                                        hand_pos=self.hand_base.xpos.copy(),
                                        hand_quat_wxyz=self.hand_base.xquat.copy())
                self.hand_opt = HandOptimizer(mj_data=self.mj_data,
                                              hand_params=hand_params,
                                              object_data=self.object_data,
                                              obstacles_data=[self.env_bodies_data] +
                                                             [self.obj_obstacle_data] if self.obj_obstacle_data else [],
                                              opt_params=self.opt_params,
                                              apply_force_closure=True,
                                              to_mano_frame=False,
                                              device=self.device)
                self.next_grasp: HandGrasp = None
                self.is_last_plan_by_mpc: bool = False

    def config_fabrics(self):
        pass

    def init_ik_controllers(self):
        # DIFF-IK
        # NOTE: MjData is created here-in if needed in robot's configuration
        self.diff_ik = ArmHandDiffIK(self.mj_model, mj.MjData(self.mj_model), self.robot_class, self.qpos_home,
                                     main_ee_name=self.robot_class.hand_item_full_name(self.robot_class.HAND_BASE_NAME),
                                     main_ee_obj_type='body')
        self.diff_ik.DT = self.mj_model.opt.timestep
        self.diff_ik.init()

        # FABRICS
        self.config_fabrics()
        if self.fabrics_env_class and self.fabrics_robot_class:
            assert self.fabrics_arm_xml and self.fabrics_hand_xml, "Fabrics arm/hand XMLs are not set!"
            self.fabrics_robot_class.BASE_POSES = self.robot_class.BASE_POSES
            self.fabrics_env = self.fabrics_env_class(arm_hand_class=self.fabrics_robot_class,
                                                      arm_xml=self.fabrics_arm_xml,
                                                      hand_xml=self.fabrics_hand_xml,
                                                      use_finger_fabrics=self.robot_class.FINGERS_IK_ENABLED,
                                                      fabric_cfg=self.fabric_cfg)
            self.fabrics_env.init()
            self.fabrics_robot = self.fabrics_env.robots_system
            self.fabrics_controller = self.fabrics_env.fabrics_controller

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
                if self.synchronous_controlling and (self.step_cnt % num_steps == 0 if num_steps > 1 else True):
                    self.plan()

                # Robot class specific GUI visualizing
                self.robot_class.visualize(viewer=mj_viewer, data=self.mj_data)

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
                if self.synchronous_controlling:
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
        assert not self.synchronous_controlling
        while True:
            start_time = time.time()
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
            self.is_last_plan_by_mpc = False
        elif self.mpcontroller:
            self.write_state_to_controller()
            self.mpcontroller.update_action()
            # end = time.perf_counter()
            # print("plan_time", end - start)
            self.fetch_nominal_control_spline()
            # self.sim.nominal_action = self.controller.action(self.sim.sim_backend.sim_time)
            # print("best action", self.sim.optimal_action)
            self.is_last_plan_by_mpc = True

    def _get_geoms_poses(self, geoms_data: dict[str, Any]) -> dict[str, Any]:
        geom_poses = {}
        geom_global_pos = np.zeros(3)
        geom_global_quat = np.zeros(4)
        for geom_name, geom_data in geoms_data.items():
            geom_spec = self.mj_spec.geom(geom_name)
            geom_body = self.mj_data.body(self.mj_model.geom(geom_name).bodyid[0])
            geom_body_pose = np.concatenate([geom_body.xpos, geom_body.xquat])
            mj.mju_mulPose(geom_global_pos, geom_global_quat,
                           geom_body_pose[:3], geom_body_pose[3:],
                           geom_spec.pos, geom_spec.quat)
            geom_poses[geom_name] = np.concatenate([geom_global_pos, geom_global_quat])
        return geom_poses

    def plan_arm_hand(self):
        # Update object data for hand-opt
        self.hand_opt.step_object()

        # Hand plan
        # Update [hand_opt]'s opt-wrist-pose to hand-base
        assert self.is_mujoco
        cur_arm_qpos = torch.tensor(self.mj_data.qpos[self.arm_qpos_ids], dtype=torch.float32, device=self.device)
        cur_hand_pos = torch.tensor(self.hand_base.xpos, dtype=torch.float32, device=self.device)
        cur_hand_quat_wxyz = torch.tensor(self.hand_base.xquat, dtype=torch.float32, device=self.device)
        cur_wrist_rot = cur_hand_quat_wxyz if self.hand_opt.use_quat \
            else roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(cur_hand_quat_wxyz))
        cur_joint_angles = torch.tensor(self.mj_data.qpos[self.hand_qpos_ids], dtype=torch.float32,
                                        device=self.device)

        grasp_pos = grasp_quat_wxyz = None
        if self.is_last_plan_by_mpc:
            print("BACK TO MPC")
            assert self.hand_opt.use_quat, "Only quat-based grasp is supported!"
            if self.USE_CURRENT_HAND_BASE_CENTRIC_GRASP:
                grasp_pos = cur_hand_pos.unsqueeze(0)
                grasp_quat_wxyz = cur_hand_quat_wxyz.unsqueeze(0)
                # joint_angles = cur_joint_angles.unsqueeze(0)
            else:
                if False:
                    perfect_relative_grasp = self.hand_opt.perfect_relative_grasp
                    if perfect_relative_grasp:
                        print("REUSE PERFECT GRASP AS INIT", self.hand_opt.cur_loss)
                        perfect_wrist_pose = torch.eye(4)
                        perfect_wrist_pose[:3, 3] = perfect_relative_grasp.base_pos.unsqueeze(0)
                        perfect_wrist_pose[:3, :3] = roma.unitquat_to_rotmat(
                            roma.quat_wxyz_to_xyzw(perfect_relative_grasp.base_quat_wxyz)).unsqueeze(0)
                        perfect_wrist_pose = self.object_data.get_object_pose(self.mj_data) @ perfect_wrist_pose
                        grasp_pos = perfect_wrist_pose[:3, 3]
                        grasp_quat_wxyz = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(perfect_wrist_pose[:3, :3]))
                        # joint_angles = perfect_relative_grasp.joint_angles
                else:
                    print("RECALCULATE INIT GRASP", self.hand_opt.cur_loss)
                    new_initial_grasp = self.hand_opt.recalculate_initial_grasp()
                    grasp_pos = torch.as_tensor(new_initial_grasp.base_pos, dtype=torch.float32,
                                                device=self.device)
                    grasp_quat_wxyz = torch.as_tensor(new_initial_grasp.base_quat_wxyz, dtype=torch.float32,
                                                      device=self.device)
                    # joint_angles = torch.as_tensor(new_initial_grasp.joint_angles, dtype=torch.float32,
                    #                               device=self.device)
            if grasp_pos is not None and grasp_quat_wxyz is not None:
                mj_data_move_mocap(self.mj_model, self.mj_data, self.robot_class.EE_TARGET_MOCAP_NAME,
                                   pos=grasp_pos.cpu().numpy(), quat=grasp_quat_wxyz.cpu().numpy())
                assert self.hand_opt.use_quat
                # NOTE: Do not update [opt_joint_angles] here-in!
                self.hand_opt.update_opt_params(wrist_pos=grasp_pos, wrist_rot_wxyz=grasp_quat_wxyz)

        # if self.hand_opt.cur_loss > self.hand_opt.LOSS_MIN_THRESHOLD:
        if True:
            hand_batches_num = self.hand_opt.nbatches
            assert hand_batches_num == 1
            dist = None
            if self.next_grasp:
                dist = self.next_grasp.distance_from(HandGrasp(base_pos=cur_hand_pos,
                                                               base_quat_wxyz=cur_hand_quat_wxyz,
                                                               joint_angles=cur_joint_angles))
            grasp_optimizing = (dist is not None) and dist < 0.15
            if not dist or grasp_optimizing:
                self.next_grasp = self.hand_opt.step_optimize(
                    cur_wrist_pos=cur_hand_pos.repeat(hand_batches_num, 1),
                    cur_wrist_rot=cur_wrist_rot.repeat(hand_batches_num, 1),
                    cur_joint_angles=cur_joint_angles.repeat(hand_batches_num, 1))
                print("LOSS", self.hand_opt.cur_loss, dist, "MPC disabled", self.sim.task.mpc_disabled)
                if (dist is not None) and dist < 0.075:
                    grasp_optimizing = False
                    self.sim.task.desired_hand_qpos = self.next_grasp.joint_angles.cpu().numpy()
            # print("grasp_optimizing", grasp_optimizing)
            self.sim.task.mpc_disabled = (self.hand_opt.cur_loss > self.hand_opt.LOSS_MIN_THRESHOLD
                                          or grasp_optimizing)

        # Arm plan
        next_grasp_pose_wxyz = self.next_grasp.base_pose_wxyz
        mj_data_move_mocap(self.mj_model, self.mj_data, self.robot_class.EE_TARGET_MOCAP_NAME,
                           pos=self.next_grasp.base_pos.cpu().numpy(),
                           quat=self.next_grasp.base_quat_wxyz.cpu().numpy())
        next_grasp_joint_angles = self.next_grasp.joint_angles.cpu().numpy()
        if self.USE_DIFF_IK:
            # NOTE: [self.diff_ik.data] != self.mj_data
            next_grasp_pose_wxyz = next_grasp_pose_wxyz.cpu().numpy()
            target_ee_poses = {self.diff_ik.main_ee_name: next_grasp_pose_wxyz}
            if self.robot_class.FINGERS_IK_ENABLED:
                for ftip_name in self.robot_class.FINGER_TIPS_NAMES:
                    # NOTE: [self.hand_opt.hand_layer.mj_data] is hand only, so using plain [ftip_name] directly!
                    target_ee_poses[self.robot_class.hand_item_full_name(ftip_name)] = (
                        mj_data_site_pose(self.hand_opt.hand_layer.mj_data, ftip_name, as_single_array=True))
            else:
                self.diff_ik.posture_task.target_q[self.hand_qpos_ids] = next_grasp_joint_angles
            q = self.diff_ik.plan(target_ee_poses=target_ee_poses, use_solver=True)
            if q is None:
                q = self.diff_ik.plan(target_ee_poses=target_ee_poses, use_solver=False)
        else:
            obj_geom_quat = np.zeros(4)
            mj.mju_mat2Quat(obj_geom_quat, self.obj_visual_geom.xmat)
            # cur_obj_poses = {
            #   self.object_data.obj_name: np.concat([self.obj_visual_geom.xpos, obj_geom_quat])}
            self.fabrics_controller.step(
                new_qpos_target=torch.cat([cur_arm_qpos, torch.as_tensor(self.next_grasp.joint_angles)]),
                new_hand_base_target=torch.as_tensor(next_grasp_pose_wxyz),
                new_finger_targets=self.hand_opt.hand_layer.fingertip_fabric_poses)
            q = self.fabrics_controller.q_prev.detach().cpu().numpy().squeeze()
        self.mj_robot_ctrl[self.arm_ctrl_ids] = q[self.arm_qpos_ids]
        self.mj_robot_ctrl[self.hand_ctrl_ids] = q[self.hand_qpos_ids]

        # Visualizing
        #
        # - Next grasp
        if False:
            self.hand_opt.visualize_grasp(self.next_grasp, self.object_data.geom_meshes)

        # - Fabrics
        if not self.USE_DIFF_IK and self.fabrics_controller and self.is_mujoco:
            self.fabrics_controller.visualize_fabrics(self.sim.mj_viewer.user_scn)

        # - Hand pcl
        self.visualize_optimized_hand_pcl()

        # - Obj pcl
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
        # TODO: In case on non-synchronous controller: Fetch state data from sim
        self.mpcontroller.update_state(self.sim.sim_state)

    def fetch_nominal_control_spline(self) -> None:
        """Util that publishes the current controller spline."""
        # Set sim's control spline from [self.controller]
        self.update_control(self.mpcontroller.nominal_spline_data)

        # Visualize traces
        if self.mpcontroller.traces is not None and len(self.mpcontroller.traces) > 0:
            pass

    @classmethod
    def _draw_points(cls, mj_scene: mj.MjvScene, points: npt.ArrayLike, size: list[float], color: list[float]):
        if isinstance(points, np.ndarray):
            assert len(points.shape) == 2
        npoints = len(points)
        mj_scene_draw_spheres(mj_scene, positions=points,
                              sizes=npoints * [size], rgbas=npoints * [color])

    def visualize_optimized_hand_pcl(self):
        if self.is_mujoco:
            mj_user_scn = self.sim.mj_viewer.user_scn
            # Hand pcl/anchors
            if self.hand_opt.hand_visual_verts is not None:
                self._draw_points(mj_user_scn, self.hand_opt.hand_visual_verts, size=[0.005], color=[1, 1, 0, 1])
            if self.hand_opt.hand_anchors is not None:
                hand_anchors = self.hand_opt.hand_anchors.squeeze().detach().cpu().numpy()
                self._draw_points(mj_user_scn, hand_anchors, size=[0.01], color=[1, 0, 0, 1])

            # Grasp site
            if self.hand_opt.hand_visual_grasp_site_pos is not None:
                self._draw_points(mj_user_scn, self.hand_opt.hand_visual_grasp_site_pos, size=[0.01],
                                  color=[0, 0, 1, 1])
            if self.hand_opt.hand_visual_grasp_direction_site_pos is not None:
                self._draw_points(mj_user_scn, self.hand_opt.hand_visual_grasp_direction_site_pos, size=[0.01],
                                  color=[1, 0, 1, 1])

            # Fingertips
            self._draw_points(mj_user_scn, [ftip_pose.detach().cpu().numpy()[:3] for _, ftip_pose in
                                            self.hand_opt.hand_layer.fingertip_fabric_poses.items()], size=[0.01],
                              color=[0.5, 0.5, 1, 1])

    def visualize_obj_pcl(self):
        if self.is_mujoco:
            obj_points = self.hand_opt.object_data.all_points.detach().cpu().numpy().tolist()
            mj_scene_draw_spheres(self.sim.mj_viewer.user_scn,
                                  positions=obj_points,
                                  sizes=len(obj_points) * [[0.005]],
                                  rgbas=len(obj_points) * [[0, 1, 0, 1]])

            if True:
                object_dense_pcl = self.hand_opt.object_global_dense_pcl.tolist()
                mj_scene_draw_spheres(self.sim.mj_viewer.user_scn,
                                      positions=object_dense_pcl,
                                      sizes=len(object_dense_pcl) * [[0.005]],
                                      rgbas=len(object_dense_pcl) * [[1, 0, 0, 1]])

            if False:
                obj_obstacle_pts = np.concat(
                    [obst.all_points.detach().cpu().numpy() for obst in self.hand_opt.obstacles_data]).tolist()
                mj_scene_draw_spheres(self.sim.mj_viewer.user_scn,
                                      positions=obj_obst_points,
                                      sizes=len(obj_obst_points) * [[0.005]],
                                      rgbas=len(obj_obst_points) * [[1, 0, 0, 1]])
