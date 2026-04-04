import trimesh
import os
import numpy as np
import trimesh.sample
import coacd
import open3d as o3d

HAND_LAYERS_DIR = os.path.dirname(os.path.abspath(__file__))
LEAP_LAYER_CACHE_DIR = f'{HAND_LAYERS_DIR}/cache/leap'


def o3d_vox_downsample(points, voxel_size=0.005):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    downsampled_pc = pc.voxel_down_sample(voxel_size)
    points = np.asarray(downsampled_pc.points)
    return points


def o3d_uniform_downsample(points, K=5):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    downsampled_pc = pc.uniform_down_sample(every_k_points=K)
    points = np.asarray(downsampled_pc.points)
    return points
