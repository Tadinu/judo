from typing import Optional
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from xvfbwrapper import Xvfb

# mjmanip
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig
from mjmanip.control.fabrics.fabrics_controller import FabricsController
from mjmanip.robot.panda_leap_fabrics import (ARM_XML_PATH as PANDA_LEAP_FABRICS_ARM_XML_PATH,
                                              HAND_XML_PATH as PANDA_LEAP_FABRICS_HAND_XML_PATH, \
                                              PandaLeapWithFabricsEnv, PandaLeapWithFabrics)

# judo
from judo import PACKAGE_ROOT
from judo.app.mpc_app import MPCApp
from judo.tasks.panda_leap_pick import PandaLeapPickConfig

CONFIGS_DIR = f"{PACKAGE_ROOT}/configs"
FABRICS_CONFIGS_DIR = f"{CONFIGS_DIR}/fabrics"
panda_leap_fabric_cfg_name = "panda_leap_mujoco"
panda_leap_judo_cfg_name = "judo_dora_panda_leap_pick"

# we store judo_dora_default in the config store so that custom dora configs outside of judo can inherit from it
cs = ConfigStore.instance()
with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base="1.3"):
    # NOTE: Don't name this directly as "judo_dora_default" so it doesn't clash
    cs.store(panda_leap_judo_cfg_name, compose(config_name=panda_leap_judo_cfg_name))
    cs.store(panda_leap_fabric_cfg_name, ArmHandPoseFabricConfig)

task_registration_cfg = optimizer_registration_cfg = None


@hydra.main(config_path=str(CONFIGS_DIR), config_name=panda_leap_judo_cfg_name, version_base="1.3")
def fetch_judo_cfgs(cfg: DictConfig) -> None:
    """Main function to run judo via a hydra configuration yaml file."""
    global task_registration_cfg, optimizer_registration_cfg
    task_registration_cfg = cfg.custom_tasks
    optimizer_registration_cfg = cfg.custom_optimizers


panda_leap_fabric_cfg = None


@hydra.main(version_base=None, config_path=FABRICS_CONFIGS_DIR, config_name=panda_leap_fabric_cfg_name)
def fetch_fabric_cfg(cfg: DictConfig) -> None:
    global panda_leap_fabric_cfg
    panda_leap_fabric_cfg = OmegaConf.to_object(cfg)
    assert isinstance(panda_leap_fabric_cfg, ArmHandPoseFabricConfig)
    # print(OmegaConf.to_yaml(panda_leap_fabric_cfg))


class PandaLeapPickApp(MPCApp):
    def __init__(self, task_registration_cfg: Optional[DictConfig] = None,
                 optimizer_registration_cfg: Optional[DictConfig] = None,
                 fabric_cfg: Optional[ArmHandPoseFabricConfig] = None,
                 headless: bool = False) -> None:
        cfg = PandaLeapPickConfig()
        cfg.robot_class.BASE_PLATFORM_NAME = "base_platform"
        super().__init__(task_name=cfg.task_name,
                         robot_class=cfg.robot_class,
                         optimizer_name=list(optimizer_registration_cfg.keys())[0],
                         sim_backend_type=cfg.sim_backend_type(),
                         task_registration_cfg=task_registration_cfg,
                         optimizer_registration_cfg=optimizer_registration_cfg,
                         fabric_cfg=fabric_cfg,
                         headless=headless)

    def config_fabrics(self):
        self.fabrics_env_class = PandaLeapWithFabricsEnv
        self.fabrics_robot_class = PandaLeapWithFabrics
        self.fabrics_arm_xml = PANDA_LEAP_FABRICS_ARM_XML_PATH
        self.fabrics_hand_xml = PANDA_LEAP_FABRICS_HAND_XML_PATH


def run_app(headless: bool) -> None:
    app = PandaLeapPickApp(task_registration_cfg=task_registration_cfg,
                           optimizer_registration_cfg=optimizer_registration_cfg,
                           fabric_cfg=panda_leap_fabric_cfg,
                           headless=headless)
    app.spin()


if __name__ == "__main__":
    fetch_judo_cfgs()
    fetch_fabric_cfg()
    headless = False
    if headless:
        with Xvfb(width=1920, height=1080) as xvfb:
            print(f"Using Xvfb display: {xvfb.new_display}")
            run_app(headless)
    else:
        run_app(headless)
