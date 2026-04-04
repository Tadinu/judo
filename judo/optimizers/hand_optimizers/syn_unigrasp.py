import os
import random
import torch
import numpy as np
import glob
import roma
import trimesh

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
from object_utils import ObjectData

BATCHES_NUM = 1


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
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
        object_data = ObjectData.get_meshes_data([obj_filepath], device=device) if obj_filepath.endswith('.obj') else (
            ObjectData.get_mj_object_data(obj_filepath, body_names=['mug'] if use_mug else None,
                                          device=device))
        # object_data.visualize()
        # obj_name = obj_filepath.split('/')[-1].split('.')[0]
        hand_opt = HandOptimizer(hand_params=HandParams.get(hand_model_name=hand_model_name,
                                                            xml_path=HAND_XML_PATH,
                                                            joint_angles=np.zeros(PANDA_LEAP.HAND_DOFS_NO,
                                                                                  dtype=np.float32),
                                                            hand_pos=np.array([0, 0, 0], dtype=np.float32),
                                                            hand_quat=np.array([0.71, 0.71, 0, 0], dtype=np.float32)),
                                 object_data=object_data,
                                 opt_params=opt_params,
                                 device=device)

        hand_opt.optimize(obstacle=None, n_iters=200)
        grasp = hand_opt.best_grasp_configuration()

        # One more step
        if False:
            new_wrist_rot = torch.tensor([0.71, 0.71, 0, 0]).repeat(BATCHES_NUM, 1) if hand_opt.use_quat \
                else roma.unitquat_to_rotmat(torch.tensor([0.71, 0.71, 0, 0])).repeat(BATCHES_NUM, 1, 1)
            grasp = hand_opt.step_optimize(cur_wrist_pos=np.tile([0, 0, 0.1], (BATCHES_NUM, 1)),
                                           cur_wrist_rot=new_wrist_rot,
                                           cur_obj_mesh_poses=[np.array([0, 0, 0, 1, 0, 0, 0])],
                                           substeps_num=200)

        # Visualize
        vis_grasp = True
        if vis_grasp:
            hand_opt.visualize_grasp(grasp, object_data.meshes)
