# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Dict, Tuple, Type

from judo.tasks.base import Task, TaskConfig
from judo.tasks.caltech_leap_cube import CaltechLeapCube, CaltechLeapCubeConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from judo.tasks.fr3_pick import FR3Pick, FR3PickConfig
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.tasks.leap_cube_down import LeapCubeDown, LeapCubeDownConfig
from judo.tasks.leap_freejoint_object_pick import LeapFreeJointObjectPick, LeapFreeJointObjectPickConfig
from judo.tasks.allegro_cube_rotate import AllegroCubeRotate, AllegroCubeRotateConfig
from judo.tasks.iiwa7_allegro_pick import IIWA7AllegroPick, IIWA7AllegroPickConfig

_registered_tasks: Dict[str, Tuple[Type[Task], Type[TaskConfig]]] = {
    CylinderPush.config_t.task_name: (CylinderPush, CylinderPushConfig),
    Cartpole.config_t.task_name: (Cartpole, CartpoleConfig),
    FR3Pick.config_t.task_name: (FR3Pick, FR3PickConfig),
    LeapCube.config_t.task_name: (LeapCube, LeapCubeConfig),
    LeapCubeDown.config_t.task_name: (LeapCubeDown, LeapCubeDownConfig),
    CaltechLeapCube.config_t.task_name: (CaltechLeapCube, CaltechLeapCubeConfig),
    AllegroCubeRotate.config_t.task_name: (AllegroCubeRotate, AllegroCubeRotateConfig),
    IIWA7AllegroPick.config_t.task_name: (AllegroCubeRotate, AllegroCubeRotateConfig),
    LeapFreeJointObjectPick.config_t.task_name: (LeapFreeJointObjectPick, LeapFreeJointObjectPickConfig),
}


def get_registered_tasks() -> Dict[str, Tuple[Type[Task], Type[TaskConfig]]]:
    """Returns a dictionary of registered tasks."""
    return _registered_tasks


def register_task(name: str, task_type: Type[Task], task_config_type: Type[TaskConfig]) -> None:
    """Registers a new task."""
    _registered_tasks[name] = (task_type, task_config_type)


__all__ = [
    "get_registered_tasks",
    "register_task",
    "Task",
    "TaskConfig",
    "CaltechLeapCube",
    "CaltechLeapCubeConfig",
    "Cartpole",
    "CartpoleConfig",
    "CylinderPush",
    "CylinderPushConfig",
    "FR3Pick",
    "FR3PickConfig",
    "LeapCube",
    "LeapCubeConfig",
    "LeapCubeDown",
    "LeapCubeDownConfig",
    "LeapFreeJointObjectPick",
    "LeapFreeJointObjectPickConfig",
    "AllegroCubeRotate",
    "AllegroCubeRotateConfig",
]
