from dataclasses import dataclass
from typing import Optional
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from pathlib import Path

# judo
from judo import PACKAGE_ROOT
from judo.tasks.leap_freejoint_object_pick import LeapFreeJointObjectPickConfig
from judo.app.mpc_app import MPCApp

# warp
import warp as wp

# mjmanip
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig

FABRICS_CONFIGS_DIR = f"{PACKAGE_ROOT}/configs/fabrics"

cs = ConfigStore.instance()
cs.store(name="leap_rh", node=ArmHandPoseFabricConfig)

cfg_name = "leap_rh"

fabric_cfg = None


@hydra.main(version_base=None, config_path=FABRICS_CONFIGS_DIR, config_name=cfg_name)
def gb_fetch_fabric_config(cfg: DictConfig) -> None:
    global fabric_cfg
    fabric_cfg = OmegaConf.to_object(cfg)
    assert isinstance(fabric_cfg, ArmHandPoseFabricConfig)
    # print(OmegaConf.to_yaml(fabric_cfg))


gb_fetch_fabric_config()


@wp.kernel
def wp_kernel_set_leap_joint_targets(
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
    # ROBOT_INIT_QUAT = wp.normalize(wp.quat(0.283, 0.683, -0.622, 0.258))
    world_id = wp.tid()
    ctrl = control[world_id]
    jid = joint_id[world_id]

    # Move finger joints
    for i in range(len(jid)):
        joint_dof_id = joint_qd_start[jid[i]]
        # wp.printf("joint_id %d %d\n", jid[i], joint_dof_id)
        joint_target_pos[joint_dof_id] = wp.clamp(ctrl[joint_dof_id],
                                                  joint_limit_lower[joint_dof_id], joint_limit_upper[joint_dof_id])

    # update the sim time
    world_time[world_id] += sim_dt


class LeapFreeJointObjectPickApp(MPCApp):
    def __init__(self,
                 task_registration_cfg: Optional[DictConfig] = None,
                 optimizer_registration_cfg: Optional[DictConfig] = None,
                 fabric_cfg: Optional[ArmHandPoseFabricConfig] = None) -> None:
        cfg = LeapFreeJointObjectPickConfig()
        super().__init__(task_name=cfg.task_name,
                         optimizer_name=list(optimizer_registration_cfg.keys())[0],
                         sim_backend_type=cfg.sim_backend_type(),
                         wp_kernel_set_joint_targets=wp_kernel_set_leap_joint_targets,
                         task_registration_cfg=task_registration_cfg,
                         optimizer_registration_cfg=optimizer_registration_cfg,
                         fabric_cfg=fabric_cfg)


task_reg_cfg = optimizer_reg_cfg = None

CONFIG_PATH = (Path(__file__).parent.parent / "configs").resolve()


@hydra.main(config_path=str(CONFIG_PATH), config_name="judo_dora_leap_freejoint_object_pick", version_base="1.3")
def fetch_cfgs(cfg: DictConfig) -> None:
    """Main function to run judo via a hydra configuration yaml file."""
    global task_reg_cfg, optimizer_reg_cfg
    task_reg_cfg = cfg.custom_tasks
    optimizer_reg_cfg = cfg.custom_optimizers
    # controller_config_overrides = cfg.controller_config_overrides
    # optimizer_config_overrides = cfg.optimizer_config_overrides


# we store judo_dora_default in the config store so that custom dora configs outside of judo can inherit from it
cs = ConfigStore.instance()
with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base="1.3"):
    default_cfg = compose(config_name="judo_dora_default")
    cs.store("judo_dora", default_cfg)  # don't name this judo_dora_default so it doesn't clash

if __name__ == "__main__":
    fetch_cfgs()
    app = LeapFreeJointObjectPickApp(task_registration_cfg=task_reg_cfg,
                                     optimizer_registration_cfg=optimizer_reg_cfg,
                                     fabric_cfg=fabric_cfg)
    app.spin()
