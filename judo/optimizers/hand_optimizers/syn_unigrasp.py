import os
import random
import torch
import numpy as np
import glob
import roma

from hand_optimizer import HandOptimizer, HandOptimizerParams, HandGrasp
from mjmanip.robot.panda_leap_mjx import PandaLeapMjxEnv, PandaLeapMjx, ARM_SCENE_XML_PATH, ARM_XML_PATH, HAND_XML_PATH

if PandaLeapMjx:
    PandaLeapMjx.NINSTANCES = 1
    PandaLeap = PandaLeapMjx
if PandaLeapMjxEnv:
    PandaLeapEnv = PandaLeapMjxEnv

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
    torch.set_default_device(torch.device('cuda:0'))
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hand_model_name = 'leap_hand'
    opt_params = HandOptimizerParams(n_batches=BATCHES_NUM, distance_lower=0.05, distance_upper=0.15,
                                     jitter_strength=0.1,
                                     joint_limit_lower=-np.pi / 6,
                                     joint_limit_upper=np.pi / 6)

    use_mug = True
    if use_mug:
        filepath_list = [
            "/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f891/11_MPC/judo/judo/models/xml/objects/mug/mug.xml"]
    else:
        mesh_dir = f'{CUR_DIR}/meshes/'
        filepath_list = glob.glob('{}/*.obj'.format(mesh_dir))

    for obj_filepath in filepath_list:
        object_data = ObjectData.get_meshes_data([obj_filepath], device=device) if obj_filepath.endswith('.obj') else (
            ObjectData.get_mj_object_data(obj_filepath, body_names=['mug'] if use_mug else None,
                                          device=device))
        obj_name = obj_filepath.split('/')[-1].split('.')[0]
        hand_opt = HandOptimizer(device=device, hand_params=HandParams(hand_model_name='leap_hand',
                                                                       xml_path=HAND_XML_PATH),
                                 object_data=object_data,
                                 opt_params=opt_params)

        hand_opt.optimize(obstacle=None, n_iters=200)

        # One more step
        if False:
            new_wrist_rot = torch.tensor([0.71, 0.71, 0, 0]).repeat(BATCHES_NUM, 1) if hand_opt.use_quat \
                else roma.unitquat_to_rotmat(torch.tensor([0.71, 0.71, 0, 0])).repeat(BATCHES_NUM, 1, 1)
            hand_opt.step_optimize(cur_wrist_pos=np.tile([0, 0, 1], (BATCHES_NUM, 1)),
                                   cur_wrist_rot=new_wrist_rot,
                                   cur_mesh_poses=[np.array([0, 0, 1, 0.71, 0.71, 0, 0])])
        grasp = hand_opt.best_grasp_configuration()

        # Visualize
        vis_grasp = True
        if vis_grasp:
            hand_opt.visualize_grasp(grasp, object_data.meshes)
