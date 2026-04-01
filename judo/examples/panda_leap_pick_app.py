from typing import Optional
from pathlib import Path
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from xvfbwrapper import Xvfb

# judo
from judo.app.mpc_app import MPCApp
from judo.tasks.panda_leap_pick import PandaLeapPickConfig


class PandaLeapPickApp(MPCApp):
    def __init__(self,
                 task_registration_cfg: Optional[DictConfig] = None,
                 optimizer_registration_cfg: Optional[DictConfig] = None,
                 headless: bool = False) -> None:
        cfg = PandaLeapPickConfig()
        cfg.robot_class.BASE_PLATFORM_NAME = "base_platform"
        super().__init__(task_name=cfg.task_name,
                         robot_class=cfg.robot_class,
                         optimizer_name=list(optimizer_registration_cfg.keys())[0],
                         sim_backend_type=cfg.sim_backend_type(),
                         task_registration_cfg=task_registration_cfg,
                         optimizer_registration_cfg=optimizer_registration_cfg,
                         headless=headless)


task_registration_cfg = optimizer_registration_cfg = None

CONFIG_PATH = (Path(__file__).parent.parent / "configs").resolve()


@hydra.main(config_path=str(CONFIG_PATH), config_name="judo_dora_panda_leap_pick", version_base="1.3")
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


def run_app(headless: bool) -> None:
    app = PandaLeapPickApp(task_registration_cfg=task_registration_cfg,
                           optimizer_registration_cfg=optimizer_registration_cfg,
                           headless=headless)
    app.spin()


if __name__ == "__main__":
    fetch_cfgs()
    headless = False
    if headless:
        with Xvfb(width=1920, height=1080) as xvfb:
            print(f"Using Xvfb display: {xvfb.new_display}")
            run_app(headless)
    else:
        run_app(headless)
