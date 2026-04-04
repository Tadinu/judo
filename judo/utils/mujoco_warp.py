import time
import numpy as np
import mujoco as mj
import mujoco_warp as mjw
import warp as wp


class MJWarpBackend:
    def __init__(self, rollout_timesteps: int = 10,
                 state_type: mj.mjtState = mj.mjtState.mjSTATE_PHYSICS):
        self.mj_model: mj.MjModel = None
        self.mjw_model: mjw.Model = None
        self.mjw_data: mjw.Data = None
        self.mj_state_type = state_type

        # Settings
        self.fps: int = 50
        self.frame_dt: float = 1.0 / self.fps
        self.wp_device = wp.get_device()

        # Rollout
        self.rollout_timesteps: int = rollout_timesteps
        self.rollout_states: list[wp.array2d(dtype=float)] = None
        self.rollout_controls: list[wp.array2d(dtype=float)] = None
        self.rollout_sensors: list[wp.array2d(dtype=float)] = None

        # Capture
        self.graph = None

    def set_model(self, mj_model: mj.MjModel, mjw_model: mjw.Model, mjw_data: mjw.Data) -> None:
        # Model evaluation
        self.mj_model = mj_model
        self.mjw_model = mjw_model
        self.mjw_data = mjw_data
        self.rollout_states = [wp.zeros((self.mjw_data.nworld, mj.mj_stateSize(mj_model, self.mj_state_type)),
                                        dtype=float)] * self.rollout_timesteps
        assert self.mjw_model.nsensordata == self.mj_model.nsensordata
        self.rollout_sensors = [wp.zeros((self.mjw_data.nworld, self.mjw_model.nsensordata),
                                         dtype=float)] * self.rollout_timesteps
        self.rollout_controls = [wp.zeros((self.mjw_data.nworld, self.mjw_model.nu),
                                          dtype=float)] * self.rollout_timesteps
        # self.world_time = wp.zeros(model_builder.world_count, dtype=wp.float32)

        # Joint ids
        # self.joint_ids = wp.array([mj_model.joint(_).id for _ in self.joint_names], dtype=wp.int32)

        # Capture (always the last, only once the model is fully setup)
        self.graph = None
        self.capture()

    def capture(self) -> None:
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._rollout_simulation()
            self.graph = capture.graph

    def _rollout_simulation(self) -> None:
        # mjw.step(self.mjw_model, self.mjw_data)
        mjw.forward(self.mjw_model, self.mjw_data)

    def rollout(self,
                state: np.ndarray,
                state_type: mj.mjtState,
                controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Prepare [self.mjw_data]
        mjw_state = wp.array(np.full((self.mjw_data.nworld, mj.mj_stateSize(self.mj_model, state_type)), state),
                             dtype=float)
        mjw.set_state(self.mjw_model, self.mjw_data, mjw_state, int(state_type))

        # shape = (num_rollouts/num_worlds, num_steps, nu)
        assert controls.shape == (self.mjw_data.nworld, self.rollout_timesteps, self.mjw_model.nu)
        self.rollout_controls = [wp.array(controls[:, i, :].squeeze(), dtype=wp.float32, device=self.wp_device)
                                 for i in range(self.rollout_timesteps)]

        # rollout
        for i in range(self.rollout_timesteps):
            self.mjw_data.ctrl.assign(self.rollout_controls[i])
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self._rollout_simulation()
            mjw.get_state(self.mjw_model, self.mjw_data, self.rollout_states[i], int(self.mj_state_type))
            self.rollout_sensors[i].assign(self.mjw_data.sensordata)
        out_states = np.array([_.numpy() for _ in self.rollout_states]).transpose(1, 0, 2)
        out_sensors = np.array([_.numpy() for _ in self.rollout_sensors]).transpose(1, 0, 2)
        return out_states, out_sensors
