import time
from typing import Callable, Optional
from threading import Lock
from omegaconf import DictConfig

import warp as wp
import mujoco as mj
from mujoco import viewer

# judo
from judo import BackendType
from judo.config import get_override_config
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.nt_simulation import NTSimulation
from judo.controller import Controller, make_controller
from judo.utils.fabrics_utils import FabricsAgent, FABRICS_MPC_TYPE
from judo.app.structs import SplineData
from judo.app.utils import get_class_from_string

# mjmanip
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig
from mjmanip.utils import mj_clear_scene, mj_draw_spheres, mj_draw_text


class MPCApp:
    def __init__(self, task_name: str,
                 optimizer_name: str,
                 sim_backend_type: BackendType,
                 kernel_set_joint_targets: Optional[Callable] = None,
                 task_registration_cfg: Optional[DictConfig] = None,
                 optimizer_registration_cfg: Optional[DictConfig] = None,
                 fabric_cfg: Optional[ArmHandPoseFabricConfig] = None) -> None:
        """Initialize the simulation node."""
        self.task_name = task_name
        self.step_cnt = 0

        # 1- Sim
        optimizer_config_cls = get_class_from_string(optimizer_registration_cfg[optimizer_name].config)
        num_rollouts = get_override_config(optimizer_config_cls, task_name)["num_rollouts"]
        match sim_backend_type:
            case BackendType.MUJOCO | BackendType.MUJOCO_WARP:
                self.sim = MJSimulation(init_task=task_name,
                                        num_rollout_worlds=num_rollouts,
                                        task_registration_cfg=task_registration_cfg)
            case BackendType.NEWTON:
                self.sim = NTSimulation(init_task=task_name,
                                        num_substeps=8,  # MPC
                                        num_rollout_worlds=num_rollouts,
                                        kernel_set_joint_targets=kernel_set_joint_targets,
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
            self.sim.controller = self.controller = make_controller(
                sim=self.sim,
                init_task=self.sim.task,
                init_optimizer=optimizer_name,
                task_registration_cfg=task_registration_cfg,
                optimizer_registration_cfg=optimizer_registration_cfg,
                rollout_backend=sim_backend_type
            )
            print("CONTROLLER NUM_TIMESTEPS", self.controller.num_timesteps)
            self.fetch_nominal_control_spline()
            self.write_state_to_controller()
            # Only for task update if from another thread
            self.lock = Lock()

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
        with mj.viewer.launch_passive(model=main_model, data=main_data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
            mj.mjv_defaultFreeCamera(main_model, viewer.cam)
            while viewer.is_running():
                mj.mj_camlight(main_model, main_data)
                if self.synchronous_controller and self.step_cnt % num_steps == 0:
                    self.write_state_to_controller()
                    self.plan()
                self.mj_visualize_traces(viewer.user_scn)
                self.sim.step()
                viewer.sync()
                self.step_cnt += 1
            viewer.close()

    def mj_visualize_traces(self, scene):
        mj_clear_scene(scene)
        # Nominal fabrics agent (the sim one, not rollout)'s sampled EE targets
        fabrics_traces = self.sim.fabrics_agent.sampled_target_traces if self.sim.fabrics_agent else None
        if fabrics_traces:
            mj_draw_spheres(scene, fabrics_traces, [0.01] * len(fabrics_traces))
        if hasattr(self.sim.task, "cur_phase"):
            mj_draw_text(scene, self.sim.task.cur_phase.name)

    def nt_spin(self):
        viewer = self.sim.sim_backend.viewer
        while viewer.is_running():
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
                print(f"Sim step {dt_elapsed:.3f} longer than desired step {dt_des:.3f}!")

        # Close viewer
        viewer.close()

    def controller_spin(self):
        """Spin logic for the controller node."""
        assert not self.synchronous_controller
        while True:
            start_time = time.time()
            # TODO:Fetch state data from sim
            self.write_state_to_controller()
            self.plan()

            # Force controller to run at fixed rate specified by control_freq.
            sleep_dt = 1 / self.controller.controller_cfg.control_freq - (time.time() - start_time)
            time.sleep(max(0, sleep_dt))

    def plan(self) -> None:
        """Updates the controls state internally."""
        if self.sim.paused:
            return

        start = time.perf_counter()
        # Rollout controller here-in!
        self.controller.update_action()
        end = time.perf_counter()

        # print("plan_time", end - start)
        self.fetch_nominal_control_spline()
        # self.sim.nominal_action = self.controller.action(self.sim.sim_backend.sim_time)
        # print("best action", self.sim.optimal_action)

    def update_control(self, nominal_spline_data: SplineData) -> None:
        """Event handler for processing controls received from controller node."""
        nominal_control = nominal_spline_data.spline()
        self.sim.update_nominal_control_spline(nominal_control)

    def update_task(self, task_name: str) -> None:
        """Updates the task type."""
        task_entry = self.controller.available_tasks.get(task_name)
        if task_entry is not None:
            self.task_name = task_name
            task_cls, _ = task_entry
            with self.lock:
                task = task_cls(self.sim)
                optimizer = self.controller.optimizer_cls(self.controller.optimizer_config_cls(), task.nu)
                self.controller = Controller(
                    controller_config=self.controller.controller_cfg,
                    task=task,
                    optimizer=optimizer,
                )
                self.fetch_nominal_control_spline()
        else:
            raise ValueError(f"Task {task_name} not found in task registry.")

    def update_optimizer(self, optimizer_name: str) -> None:
        """Updates the optimizer type."""
        optimizer_entry = self.controller.available_optimizers.get(optimizer_name)
        if optimizer_entry is not None:
            optimizer_cls, optimizer_config_cls = optimizer_entry
            optimizer_config = optimizer_config_cls()
            optimizer = optimizer_cls(optimizer_config, self.controller.task.nu)
            with self.lock:
                self.controller.optimizer = optimizer
        else:
            raise ValueError(f"Optimizer {optimizer_name} not found in optimizer registry.")

    def reset_task(self, event: dict) -> None:
        """Resets the task."""
        with self.lock:
            self.controller.reset()
            self.fetch_nominal_control_spline()

    def toggle_paused_status(self) -> None:
        """Event handler for processing pause status updates."""
        self.sim.paused = not self.sim.paused

    def write_state_to_controller(self) -> None:
        self.controller.update_state(self.sim.sim_state)

    def fetch_nominal_control_spline(self) -> None:
        """Util that publishes the current controller spline."""
        # Set sim's control spline from [self.controller]
        self.update_control(self.controller.nominal_spline_data)

        # Visualize traces
        if self.controller.traces is not None and len(self.controller.traces) > 0:
            pass
