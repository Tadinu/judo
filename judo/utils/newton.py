from typing import Any

import numpy as np
import newton
import warp as wp

# judo
from judo.utils.warp import wp_kernel_default_set_joint_targets


def nt_copy_state(target: newton.State, source: newton.State) -> None:
    with wp.ScopedDevice(source.body_q.device):
        target.assign(source)


def nt_copy_struct(target: Any, source: Any) -> None:
    attributes = set(target.__dict__).union(source.__dict__)

    for attr in attributes:
        val_target = getattr(target, attr, None)
        val_source = getattr(source, attr, None)

        if val_target is None and val_source is None:
            continue

        array_target = isinstance(val_target, wp.array)
        array_source = isinstance(val_source, wp.array)

        if not array_target and not array_source:
            continue

        if val_target is None or not array_target:
            raise ValueError(f"Struct is missing array for '{attr}' which is present in the other state.")

        if val_source is None or not array_source:
            raise ValueError(f"Other Struct is missing array for '{attr}' which is present in this state.")

        val_target.assign(val_source)


def nt_copy_control(target: newton.Control, source: newton.Control) -> None:
    """
    Return a copy of array attributes of another newton.Control object

    Args:
        source: The source newton.Control object to copy from.

    Raises:
        ValueError: If the controls have mismatched attributes (one has an array where the other is None).
    """
    with wp.ScopedDevice(source.joint_target_pos.device):
        nt_copy_struct(target, source)


def nt_copy_contacts(target: newton.Contacts, source: newton.Contacts) -> None:
    with wp.ScopedDevice(source.rigid_contact_count.device):
        nt_copy_struct(target, source)


class NewtonBackend:
    def __init__(self, joint_names: list[str],
                 num_substeps: int = 8,  # NOTE: Num of substeps must be large enough for sim to warm up
                 for_rollout: bool = False, rollout_timesteps: int = 10,
                 headless: bool = True):
        # Settings
        self.fps: int = 50
        self.frame_dt: float = 1.0 / self.fps

        assert num_substeps > 1, f"Newton backend substeps should be large enough (>1) for the warmup!"
        self.sim_time: float = 0.0
        self.sim_substeps: int = num_substeps
        self.sim_dt: float = self.frame_dt / self.sim_substeps
        self.wp_device = wp.get_device()

        # Rollout
        self.for_rollout = for_rollout
        self.wp_for_rollout = wp.ones(1, dtype=int) if self.for_rollout else wp.zeros(1, dtype=int)
        self.rollout_timesteps: int = rollout_timesteps

        # Model
        self.model: newton.Model = None
        self.model_builder: newton.ModelBuilder = None
        self.num_steps = self.rollout_timesteps if self.for_rollout else self.sim_substeps

        # Robot
        self.joint_names: list[str] = joint_names
        self.joint_local_ids: list[int] = []
        self.joint_ids: wp.array = None
        self.joint_target_controls: wp.array = None
        self.kernel_set_joint_targets = wp_kernel_default_set_joint_targets

        # Solver
        self.solver = None

        # States
        self.state_0: newton.State = None
        self.state_1: newton.State = None
        self.rollout_states: list[newton.State] = None
        self.model_ctrl: newton.Control = None
        self.rollout_model_ctrls: list[newton.Control] = None
        self.contacts: newton.Contacts = None
        self.rollout_contacts: list[newton.Contacts] = None
        self.world_time: wp.array = None

        # Viewer
        self.headless = headless
        self.viewer = None

        # Capture
        self.graph = None

    def set_model(self, model: newton.Model, model_builder: newton.ModelBuilder) -> None:
        # Model evaluation
        self.model = model
        self.model_builder = model_builder
        self.world_time = wp.zeros(model_builder.num_worlds, dtype=wp.float32)

        # Joint ids
        joints_num = model_builder.joint_count // model_builder.num_worlds
        self.joint_local_ids = [model_builder.joint_key.index(jname) for jname in self.joint_names]
        joint_ids = np.zeros((model_builder.num_worlds, len(self.joint_names)), dtype=int)
        for world_id in range(model_builder.num_worlds):
            joint_id_offset = world_id * joints_num
            joint_ids[world_id] = [i + joint_id_offset for i in self.joint_local_ids]
        self.joint_ids = wp.array(joint_ids, dtype=wp.int32)

        # Joint target controls
        self.joint_target_controls = wp.zeros((model_builder.num_worlds, model_builder.joint_dof_count),
                                              dtype=wp.float32)

        # Eval model fk
        self.state_0 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)

        # Model solver
        self.solver = newton.solvers.SolverMuJoCo(model,
                                                  solver="newton",
                                                  integrator="implicitfast",
                                                  njmax=200,
                                                  nconmax=150,
                                                  impratio=10.0,
                                                  cone="elliptic",
                                                  iterations=100,
                                                  ls_iterations=50,
                                                  use_mujoco_cpu=False)

        # States
        self.state_1 = model.state()
        self.model_ctrl = model.control()
        self.contacts = model.collide(self.state_0)
        self.rollout_states = [model.state()] * self.num_steps
        self.rollout_contacts = [model.collide(self.state_0)] * self.num_steps
        self.rollout_model_ctrls = [model.control()] * self.num_steps

        # Viewer
        # NOTE: Thought ViewerGL has a `headless` param, but it seems it still causes trouble to the headful one.
        # -> Disable for now. TODO: Find out why!
        self.viewer = newton.viewer.ViewerGL() if not self.headless else None
        if self.viewer:
            self.viewer.set_model(model)

        # Capture (always the last, only once the model is fully setup)
        self.graph = None
        self.capture()

    def capture(self) -> None:
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                if not self.for_rollout:
                    self._prepare_model_ctrl()
                self._step_simulation()
            self.graph = capture.graph

    def _prepare_model_ctrl(self) -> None:
        with wp.ScopedDevice(self.wp_device):
            num_worlds = self.model_builder.num_worlds
            wp.launch(
                self.kernel_set_joint_targets,
                dim=num_worlds,
                inputs=[
                    self.joint_target_controls,
                    self.joint_ids,
                    self.model.joint_qd_start,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.world_time,
                    self.sim_dt,
                ],
                outputs=[self.model_ctrl.joint_target_pos, self.model.joint_X_p],
            )

    def _step_simulation(self) -> None:
        self.contacts = self.model.collide(self.state_0)
        for i in range(self.num_steps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            # NOTE: [self.viewer] does not change at run time, so no need for [wp.capture_if] here
            if self.viewer:
                self.viewer.apply_forces(self.state_0)

            # NOTE: Actually, [self.for_rollout/wp_for_rollout] also does not change at run time,
            # so capture_if is only for warp syntax demo purpose!
            wp.capture_if(self.wp_for_rollout,
                          # For rollout, [rollout_model_ctrls] has been prepared in advance by the controller
                          on_true=lambda idx: nt_copy_control(self.model_ctrl, self.rollout_model_ctrls[idx]),
                          # For sim, [model_ctrl] is set directly earlier by the sim itself
                          # on_false=lambda idx: None,
                          idx=i)

            # update the solver since we have updated the joint parent transforms
            self.solver.notify_model_changed(newton.solvers.SolverNotifyFlags.JOINT_PROPERTIES)

            # Contacts
            self.solver.step(self.state_0, self.state_1, self.model_ctrl, self.contacts, self.sim_dt)
            wp.capture_if(self.wp_for_rollout,
                          on_true=lambda idx: nt_copy_contacts(self.rollout_contacts[i], self.contacts),
                          # on_false=lambda idx: None,
                          idx=i)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.capture_if(self.wp_for_rollout,
                          on_true=lambda idx: nt_copy_state(self.rollout_states[i], self.state_0),
                          # on_false=lambda idx: None,
                          idx=i)

    def step(self):
        """Step the backend"""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._step_simulation()

        self.sim_time += self.frame_dt

    def reset(self):
        nt_copy_control(self.model_ctrl, self.model.control())
        nt_copy_contacts(self.contacts, self.model.collide(self.state_0))
        self.solver.step(self.state_0, self.state_1, self.model_ctrl, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def spin(self) -> None:
        while self.viewer.is_running():
            if not self.viewer.is_paused():
                with wp.ScopedTimer("step", active=False):
                    self.step()

            with wp.ScopedTimer("render", active=False):
                self.render()

        # Close viewer
        self.viewer.close()
