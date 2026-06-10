import os
import random
import torch
import numpy as np
import glob
import roma

from judo import MODEL_PATH
from judo.tasks.panda_leap_pick import USE_LEAP_MJX

if USE_LEAP_MJX:
    # NOTE: LeapMjx hand is more robust than Leap, so use [panda_leap_mjx] for now!
    from mjmanip.robot.panda_leap_mjx import PandaLeapMjx, ARM_SCENE_XML_PATH, ARM_XML_PATH, \
        HAND_XML_PATH

    PANDA_LEAP = PandaLeapMjx
else:
    from mjmanip.robot.panda_leap import PandaLeap, ARM_SCENE_XML_PATH, ARM_XML_PATH, HAND_XML_PATH

    PANDA_LEAP = PandaLeap
PANDA_LEAP.NINSTANCES = 1

# hand optimizer
from hand_optimizer import HandOptimizer, HandOptimizerParams, HandParams

# mjmanip
from mjmanip import OBJECT_MODELS_DIR as MJMANIP_OBJECT_MODELS_DIR
from mjmanip.entity_utils import EntityData

BATCHES_NUM = 1


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    from pathlib import Path

    set_seed(0)
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)

    hand_model_name = 'leap_hand'
    opt_params = HandOptimizerParams(nbatches=BATCHES_NUM, distance_lower=0.05, distance_upper=0.15,
                                     jitter_strength=0.1,
                                     joint_limit_lower=-np.pi / 6,
                                     joint_limit_upper=np.pi / 6)

    use_mug = True
    if use_mug:
        filepath_list = [f"{MODEL_PATH}/xml/objects/mug/mug.xml"]
    else:
        mesh_dir = f'{CUR_DIR}/meshes/'
        filepath_list = glob.glob('{}/*.obj'.format(mesh_dir))

    for obj_filepath in filepath_list:
        obj_name = Path(obj_filepath).stem
        obj_geom_names = []  # ['mug_handle0', 'mug_handle1', 'mug_handle2', 'mug_handle3']
        obj_data = EntityData.get_geoms_data(obj_name, {obj_name: obj_filepath}, device=device) \
            if obj_filepath.endswith('.obj') else (
            EntityData.get_mj_entity_data(obj_name, obj_filepath, body_names=['mug'] if use_mug else None,
                                          geom_names=obj_geom_names,
                                          meshdir=str(os.path.join(MJMANIP_OBJECT_MODELS_DIR, obj_name)),
                                          is_collision=bool(obj_geom_names),
                                          merging_meshes=False,
                                          device=device))
        # obj_data.visualize()
        # obj_name = obj_filepath.split('/')[-1].split('.')[0]
        hand_opt = HandOptimizer(hand_params=HandParams.get(hand_model_name=hand_model_name,
                                                            xml_path=HAND_XML_PATH,
                                                            joint_angles=np.zeros(PANDA_LEAP.HAND_DOFS_NO,
                                                                                  dtype=np.float32),
                                                            hand_pos=np.zeros(3),
                                                            hand_quat_wxyz=np.array([0.71, 0.71, 0, 0]),
                                                            nbatches=BATCHES_NUM),
                                 object_data=obj_data,
                                 opt_params=opt_params,
                                 device=device)

        hand_opt.optimize(obstacles=None, n_iters=200)
        grasp = hand_opt.best_grasp_configuration()

        # One more step
        if False:
            new_wrist_rot = torch.tensor([0.71, 0.71, 0, 0]).repeat(BATCHES_NUM, 1) if hand_opt.use_quat \
                else roma.unitquat_to_rotmat(torch.tensor([0.71, 0.71, 0, 0])).repeat(BATCHES_NUM, 1, 1)
            grasp = hand_opt.step_optimize(cur_wrist_pos=torch.tensor([0, 0, 0.1]).repeat(BATCHES_NUM, 1),
                                           cur_wrist_rot=new_wrist_rot,
                                           substeps_num=200)

        # Visualize
        vis_grasp = True
        if vis_grasp:
            hand_opt.visualize_grasp(grasp, obj_data.geom_trimeshes)
