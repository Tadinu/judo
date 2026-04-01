from typing import Optional, Union
from pathlib import Path
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import numpy as np

# warp
import warp as wp

# judo
from judo import BackendType
from judo.tasks.allegro_cube_rotate import AllegroCubeRotate, AllegroCubeRotateConfig
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.nt_simulation import NTSimulation

hand_rotation = wp.normalize(wp.quat(0.283, 0.683, -0.622, 0.258))


@wp.kernel
def wp_kernel_move_hand_wrist(
        control: wp.array(dtype=wp.float32, ndim=2),
        joint_id: wp.array(dtype=wp.int32, ndim=2),
        joint_qd_start: wp.array(dtype=wp.int32),
        joint_limit_lower: wp.array(dtype=wp.float32),
        joint_limit_upper: wp.array(dtype=wp.float32),
        world_time: wp.array(dtype=wp.float32),
        sim_dt: float,
        # outputs
        joint_target_pos: wp.array(dtype=wp.float32),
        joint_parent_xform: wp.array(dtype=wp.transform)
):
    world_id = wp.tid()
    root_joint_id = world_id * 23
    t = world_time[world_id]
    ctrl = control[world_id]
    jid = joint_id[world_id]

    # Move finger joints
    for i in range(len(jid)):
        joint_dof_id = joint_qd_start[jid[i]]
        # wp.printf("joint_id %d %d\n", jid[i], joint_dof_id)
        joint_target_pos[joint_dof_id] = wp.clamp(ctrl[joint_dof_id],
                                                  joint_limit_lower[joint_dof_id], joint_limit_upper[joint_dof_id])

    # Move the root (wrist) joint
    q = wp.quat_identity()
    q *= wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(t) * 0.1)
    q *= wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -t * 0.02)
    root_xform = joint_parent_xform[root_joint_id]
    joint_parent_xform[root_joint_id] = wp.transform(root_xform.p, q * hand_rotation)

    # update the sim time
    world_time[world_id] += sim_dt


class AllegroCubeMonkeySim(NTSimulation):
    def __init__(self, num_instances: int = 1,
                 task_registration_cfg: Optional[DictConfig] = None):
        assert AllegroCubeRotateConfig().sim_backend_type() == BackendType.NEWTON
        super().__init__(init_task=AllegroCubeRotateConfig().task_name,
                         is_monkey_sim=True,
                         num_rollout_worlds=num_instances,
                         wp_kernel_set_joint_targets=wp_kernel_move_hand_wrist,
                         task_registration_cfg=task_registration_cfg)

    def setup_joint_targets(self):
        world_time = self.sim_backend.world_time.numpy()
        joint_limit_lower = self.model_builder.joint_limit_lower
        joint_limit_upper = self.model_builder.joint_limit_upper
        joint_qd_start = self.model_builder.joint_qd_start
        joint_target_pos = self.sim_backend.joint_target_controls.numpy()
        joint_ids = self.sim_backend.joint_ids.numpy()
        for world_id in range(self.sim_backend.model_builder.num_worlds):
            # Randomize joints based on [world_time]
            t = world_time[world_id]
            for i in joint_ids[world_id]:
                joint_dof_id = joint_qd_start[i]
                target = np.sin(t + float(i * 6) * 0.1) * 0.15 + 0.3
                joint_target_pos[world_id][joint_dof_id] = np.clip(target, joint_limit_lower[joint_dof_id],
                                                                   joint_limit_upper[joint_dof_id])

        self.sim_backend.joint_target_controls.assign(wp.array(joint_target_pos))


task_reg_cfg = None

CONFIG_PATH = (Path(__file__).parent.parent / "configs").resolve()


@hydra.main(config_path=str(CONFIG_PATH), config_name="judo_dora_allegro_cube", version_base="1.3")
def fetch_cfgs(cfg: DictConfig) -> None:
    """Main function to run judo via a hydra configuration yaml file."""
    global task_reg_cfg
    task_reg_cfg = cfg.custom_tasks


# we store judo_dora_default in the config store so that custom dora configs outside of judo can inherit from it
cs = ConfigStore.instance()
with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base="1.3"):
    default_cfg = compose(config_name="judo_dora_default")
    cs.store("judo_dora", default_cfg)  # don't name this judo_dora_default so it doesn't clash

if __name__ == "__main__":
    fetch_cfgs()
    app = AllegroCubeMonkeySim(num_instances=5, task_registration_cfg=task_reg_cfg)
    app.spin()
