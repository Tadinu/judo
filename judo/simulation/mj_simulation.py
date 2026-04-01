# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import os
from typing import Optional, Union
from mujoco import mj_step
from omegaconf import DictConfig

import numpy as np
import mujoco as mj

# judo
from judo import PACKAGE_ROOT, BackendType
from judo.app.structs import MujocoState
from judo.simulation.base import Simulation
from judo.utils.video import VideoRecorder

# mjmanip
from mjmanip.utils import mj_clear_scene, mj_draw_spheres, mj_draw_text


class MJSimulation(Simulation):
    """Mujoco simulation object.

    This class contains the data required to run a Mujoco simulation. This includes configurations, a control spline,
    and task information.

    Middleware nodes should instantiate this class and implement methods to send, process, and receive data.
    """

    def __init__(
            self,
            init_task: str = "cylinder_push",
            num_rollout_worlds: int = 1,
            task_registration_cfg: Optional[DictConfig] = None,
            kinematics_mode: bool = False,
            headless: bool = False,
            record_video: bool = False,
            width: int = 720,
            height: int = 480
    ) -> None:
        """Initialize the simulation node."""
        super().__init__(init_task=init_task, num_rollout_worlds=num_rollout_worlds,
                         task_registration_cfg=task_registration_cfg,
                         kinematics_mode=kinematics_mode,
                         headless=headless)
        if headless:
            os.environ["MUJOCO_GL"] = "egl"
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            assert os.environ["DISPLAY"] is not None, "Xvfb is required to start in advance!\n"
            "Please use xvfbwrapper. DON'T RUN: `Xvfb :<no> -screen 0 720x480x24` DIRECTLY!"
            assert record_video, "Video recording should be enabled in headless mode!"

        # Initialize video recording if enabled
        self.mj_viewer: mj.viewer.Handle = None
        self.mj_renderer: mj.Renderer = None
        self.mj_recorder: VideoRecorder = None
        if record_video:
            # Create the video recorder
            self.mj_recorder = VideoRecorder(
                output_dir=os.path.join(PACKAGE_ROOT, "recordings"),
                width=width,
                height=height,
                fps=1 / self.timestep,
            )
            # Ensure model visual offscreen buffer is compatible with video recording
            vis_global = self.task.mj_sim_model.vis.global_
            vis_global.offwidth = width
            vis_global.offheight = height
            assert self.mj_recorder.start()
            self.mj_renderer = mj.Renderer(self.task.mj_sim_model, width=width, height=height)

        # Warm up to get stabilized sim initial state
        self.step()

    def _full_step(self):
        self.task.pre_sim_step()
        if self.kinematics_mode:
            model, data = self.task.mj_sim_model, self.task.mj_data
            mj.mj_fwdPosition(model, data)
            mj.mj_comPos(model, data)
            mj.mj_sensorPos(model, data)
            # mj.mj_forward(model, data)
        else:
            mj_step(self.task.mj_sim_model, self.task.mj_data)
        self.task.post_sim_step()

        # Record frames
        if self.mj_viewer:
            self.mj_viewer.sync()
            if self.mj_renderer and self.mj_recorder and self.mj_recorder.is_recording:
                self.mj_renderer.update_scene(self.task.mj_data, self.mj_viewer.cam)
                frame = self.mj_renderer.render()
                self.mj_recorder.add_frame(frame.tobytes())

    def step(self) -> None:
        """Step the simulation forward by one timestep."""
        if self.nominal_control_spline is not None and not self.paused:
            try:
                nominal_ctrl = self.nominal_control_spline(self.task.mj_data.time)
                if self.task.map_controls:
                    nominal_ctrl = self.task.map_controls(None, nominal_ctrl[None, None, ...],
                                                          self.sim_state.data.copy()).squeeze()
                if not np.isnan(nominal_ctrl).any():
                    self.task.mj_data.ctrl[:self.task.mj_sim_model.nu] = nominal_ctrl[:self.task.mj_sim_model.nu]
            except ValueError:
                # we're switching tasks and the new task has a different number of actuators
                pass

        # Full step
        self._full_step()

        # Visualizing traces
        if self.mj_viewer:
            # NOTE: Visualizing must be done after view sync, which must have been done in `_full_step()`
            self.visualize_traces(self.mj_viewer.user_scn)

    def visualize_traces(self, scene):
        mj_clear_scene(scene)
        # Nominal fabrics agent (the sim one, not rollout)'s sampled EE targets
        traces = self.fabrics_agent.optimal_target_traces if self.fabrics_agent \
            else self.task.optimal_target_traces
        if traces:
            mj_draw_spheres(scene, traces, [0.01] * len(traces))
        if hasattr(self.task, "cur_phase"):
            mj_draw_text(scene, self.task.cur_phase.name)

    @property
    def sim_state(self) -> Union[MujocoState, np.ndarray]:
        """Returns the current simulation state."""

        # Get current task's sim state data
        # Ref: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/rollout.ipynb#scrollTo=082482c7&line=3&uniqifier=1
        state_type = self.task.mj_state_type
        current_state_data = np.zeros((mj.mj_stateSize(self.task.mj_sim_model, state_type),))
        mj.mj_getState(self.task.mj_sim_model, self.task.mj_data, current_state_data, state_type)

        # Return the right one for the backend
        backend_type = self.task.config.sim_backend_type()
        if backend_type == BackendType.MUJOCO_WARP:
            return current_state_data
        else:
            assert backend_type == BackendType.MUJOCO
            return MujocoState(time=self.task.mj_data.time,
                               data=current_state_data,
                               sim_metadata=self.task.get_sim_metadata())

    @property
    def timestep(self) -> float:
        """Returns the simulation timestep."""
        return self.task.mj_sim_model.opt.timestep

    def close(self):
        if self.mj_recorder:
            self.mj_recorder.stop()
        if self.mj_viewer:
            self.mj_viewer.close()
