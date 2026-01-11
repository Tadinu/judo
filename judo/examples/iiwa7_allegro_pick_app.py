from typing import Optional
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from pathlib import Path
from dataclasses import dataclass
from judo.tasks.iiwa7_allegro_pick import IIWA7AllegroPickConfig

# judo
from judo.app.mpc_app import MPCApp

# warp
import warp as wp


@dataclass
class IIWA7AllegroPickConfig(IIWA7AllegroPickConfig):
    """Reward configuration IIWA7-ALLEGRO obj picking task."""

    w_pos: float = 100.0
    w_rot: float = 0.1


@wp.kernel
def kernel_set_iiwa7_allegro_joint_targets(
        robot_dofs_num: int,
        control: wp.array(dtype=wp.float32, ndim=2),
        joint_qd_start: wp.array(dtype=wp.int32),
        joint_limit_lower: wp.array(dtype=wp.float32),
        joint_limit_upper: wp.array(dtype=wp.float32),
        world_time: wp.array(dtype=wp.float32),
        sim_dt: float,
        # outputs
        joint_target_pos: wp.array(dtype=wp.float32),
        joint_parent_xform: wp.array(dtype=wp.transform),
):
    ROBOT_INIT_QUAT = wp.normalize(wp.quat(0.283, 0.683, -0.622, 0.258))
    world_id = wp.tid()
    root_joint_id = world_id * 22  # robot_dofs_num
    t = world_time[world_id]
    ctrl = control[world_id]
    root_dof_start = joint_qd_start[root_joint_id]

    # Move finger joints
    for i in range(20):
        di = root_dof_start + i
        joint_target_pos[di] = wp.clamp(ctrl[di], joint_limit_lower[di], joint_limit_upper[di])

    # Move root joint transform
    q = wp.quat_identity()
    q *= wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(t) * 0.1)  # Rot-X
    q *= wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -t * 0.02)  # Rot-Z
    root_xform = joint_parent_xform[root_joint_id]
    joint_parent_xform[root_joint_id] = wp.transform(root_xform.p, q * ROBOT_INIT_QUAT)

    # update the sim time
    world_time[world_id] += sim_dt


class IIWA7AllegroPickApp(MPCApp):
    def __init__(self,
                 task_registration_cfg: Optional[DictConfig] = None,
                 optimizer_registration_cfg: Optional[DictConfig] = None) -> None:
        cfg = IIWA7AllegroPickConfig()
        super().__init__(task_name=cfg.task_name,
                         optimizer_name=list(optimizer_registration_cfg.keys())[0],
                         sim_backend_type=cfg.sim_backend_type(),
                         kernel_set_robot_joint_targets=kernel_set_iiwa7_allegro_joint_targets,
                         task_registration_cfg=task_registration_cfg,
                         optimizer_registration_cfg=optimizer_registration_cfg)


task_registration_cfg = optimizer_registration_cfg = None

CONFIG_PATH = (Path(__file__).parent.parent / "configs").resolve()


@hydra.main(config_path=str(CONFIG_PATH), config_name="judo_dora_iiwa7_allegro_pick", version_base="1.3")
def fetch_cfgs(cfg: DictConfig) -> None:
    """Main function to run judo via a hydra configuration yaml file."""
    global task_registration_cfg, optimizer_registration_cfg
    task_registration_cfg = cfg.custom_tasks
    optimizer_registration_cfg = cfg.custom_optimizers


# we store judo_dora_default in the config store so that custom dora configs outside of judo can inherit from it
cs = ConfigStore.instance()
with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base="1.3"):
    default_cfg = compose(config_name="judo_dora_default")
    cs.store("judo_dora", default_cfg)  # don't name this judo_dora_default so it doesn't clash

if __name__ == "__main__":
    fetch_cfgs()
    app = IIWA7AllegroPickApp(task_registration_cfg=task_registration_cfg,
                              optimizer_registration_cfg=optimizer_registration_cfg)
    app.spin()
