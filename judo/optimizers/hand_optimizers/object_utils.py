from __future__ import annotations
import os
from typing import Optional, Union
from dataclasses import dataclass
import trimesh
import numpy as np
import point_cloud_utils as pcu
from mesh_to_sdf import get_surface_point_cloud
from scipy.spatial import KDTree
import torch

import mujoco as mj

# mjmanip
from mjmanip.utils import IDENTITY_POSE
from mjmanip.trimesh_utils import mj_get_body_trimeshes


@dataclass
class ObjectData:
    points: list[torch.Tensor]
    normals: list[torch.Tensor]
    meshes: list[trimesh.Trimesh]
    poses: list[np.ndarray]
    mesh_filepath: str = None
    scale: float = 1.0
    mesh_paths: Optional[list[str]] = None

    @classmethod
    def get_meshes_data(cls, mesh_paths: list[str], voxel_size=0.006, scale=1.0, vis=False, watertight_process=True,
                        device: torch.device = torch.device('cuda'),
                        **kwargs) -> ObjectData:
        meshes = []
        points = []
        normals = []
        for mesh_path in mesh_paths:
            mesh = trimesh.load_mesh(mesh_path)
            meshes.append(mesh)
            # scale the object mesh
            mesh.vertices *= scale

            # Sample mesh with [watertight_process]
            v_sampled, n_sampled = get_watertight_vertices_normals(mesh, voxel_size, watertight_process,
                                                                   mesh_path, vis)
            points.append(torch.tensor(v_sampled, dtype=torch.float32, device=device))
            normals.append(torch.tensor(n_sampled, dtype=torch.float32, device=device))

        return ObjectData(points=points,
                          normals=normals,
                          mesh_paths=mesh_paths,
                          meshes=meshes,  # scaled mesh
                          poses=[IDENTITY_POSE] * len(meshes),
                          scale=scale)

    @classmethod
    def get_mj_object_data(cls, mj_model: Union[mj.MjModel, str], mj_data: Optional[mj.MjData] = None,
                           body_names: Optional[list[str]] = None,
                           voxel_size: float = 0.006, scale: float = 1.0,
                           merging_meshes: bool = True,
                           device: torch.device = torch.device('cuda')) -> ObjectData:
        if isinstance(mj_model, str):
            assert mj_model.endswith('.xml')
            mj_model: mj.MjModel = mj.MjModel.from_xml_path(mj_model)
        else:
            pass

        if not body_names:
            body_names = [mj_model.body(i).name for i in range(mj_model.nbody)]
            body_names.remove('world')
        meshes = []
        mesh_poses = []
        for _, m in mj_get_body_trimeshes(mj_model, mj_data, body_names=body_names, is_collision=True).items():
            meshes.append(m[0])
            mesh_poses.append(m[1])
        if merging_meshes:
            meshes = [trimesh.util.concatenate(meshes)]
            base_body = mj_model.body(body_names[0])
            base_body_data = mj_data.body(body_names[0]) if mj_data else None
            mesh_poses = [np.concat([base_body_data.xpos, base_body_data.xquat]) if base_body_data \
                              else np.concat([base_body.pos, base_body.quat])]
        points = []
        normals = []
        for mesh in meshes:
            # scale the object mesh
            mesh.vertices *= scale

            # Sample mesh with [watertight_process]
            v_sampled, n_sampled = get_watertight_vertices_normals(mesh, voxel_size, watertight_process=True)
            points.append(torch.tensor(v_sampled, dtype=torch.float32, device=device))
            normals.append(torch.tensor(n_sampled, dtype=torch.float32, device=device))

        return ObjectData(points=points,
                          normals=normals,
                          mesh_paths=None,
                          meshes=meshes,  # already scaled meshes
                          poses=mesh_poses,
                          scale=scale)

    @property
    def all_points(self) -> torch.Tensor:
        return torch.cat(self.points)

    @property
    def all_normals(self) -> torch.Tensor:
        return torch.cat(self.normals)

    def transform_to(self, new_poses: list[np.ndarray], resample: bool = False):
        assert len(new_poses) == len(self.meshes)
        device = self.points[0].device
        for i, mesh in enumerate(self.meshes):
            # Old mesh pose
            old_mesh_pose = self.poses[i]
            old_mesh_transf = trimesh.transformations.quaternion_matrix(old_mesh_pose[3:])
            old_mesh_transf[:3, 3] = old_mesh_pose[:3]

            # New mesh pose
            new_mesh_pose = new_poses[i]
            new_mesh_transf = trimesh.transformations.quaternion_matrix(new_mesh_pose[3:])
            new_mesh_transf[:3, 3] = new_mesh_pose[:3]

            # Apply [new_pose] to mesh in place
            delta_tf = new_mesh_transf @ np.linalg.inv(old_mesh_transf)

            # Resample the mesh
            # NOTE: points & normals dtype are torch.float32 or float32 for convenient operations later
            if resample:
                mesh.apply_transform(delta_tf)
                v_sampled, n_sampled = sample_mesh_surface(mesh)
                # NOTE: Points, normals can have new lengths so not copying here
                self.points[i] = torch.tensor(v_sampled, dtype=torch.float32, device=device)
                self.normals[i] = torch.tensor(n_sampled, dtype=torch.float32, device=device)
            else:
                delta_tf_torch = torch.tensor(delta_tf, dtype=torch.float32, device=device)
                R = delta_tf_torch[:3, :3]
                t = delta_tf_torch[:3, 3]
                self.points[i].copy_(self.points[i] @ R.T + t)
                self.normals[i].copy_(torch.nn.functional.normalize(self.normals[i] @ R.T, dim=-1))
        self.poses = new_poses


def get_stable_pose(mesh):
    # NOTE: watertight object mesh is needed for compute stable pose
    if mesh.is_watertight:
        pass
    else:
        pitch = mesh.extents.max() / 128  # size
        if pitch < 0.002:
            pitch = 0.002
        vox = mesh.voxelized(pitch)
        vox.fill()
        bounds = vox.bounds

        mesh = vox.marching_cubes
        mesh.vertices -= mesh.bounds[0]
        mesh.vertices *= pitch
        mesh.vertices += bounds[0]

    poses = mesh.compute_stable_poses(n_samples=1)[0]
    return poses


def get_watertight_vertices_normals(mesh: trimesh.Trimesh, voxel_size: float,
                                    watertight_process: bool = True,
                                    mesh_filepath: Optional[str] = None,
                                    vis: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if watertight_process:
        if mesh.is_watertight:
            pass
        else:
            pitch = max(mesh.extents.max() / 128, 0.002)
            use_binvox = mesh.faces.shape[0] > 100
            if use_binvox:
                # change it to binvox method for better and speed up
                vox = mesh.voxelized(pitch, 'binvox')
            else:
                vox = mesh.voxelized(pitch)
            vox.fill()

            mesh = vox.marching_cubes
            mesh.vertices -= mesh.bounds[0]
            if use_binvox:
                offset = 0  # 0.5
                mesh.vertices += offset
            mesh.vertices *= pitch
            mesh.vertices += vox.bounds[0]
            # vw, fw = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=50000)
            # mesh = trimesh.Trimesh(vertices=vw, faces=fw)

    v_sampled, n_sampled = sample_mesh_surface(mesh, voxel_size)
    if vis:
        print('points shape is :', v_sampled.shape)
        import open3d as o3d
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(v_sampled)
        o3d_pc.normals = o3d.utility.Vector3dVector(n_sampled)
        o3d_mesh = mesh.as_open3d
        o3d_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([o3d_pc, o3d_mesh])

    write_ply = False
    if write_ply:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(v_sampled)
        CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        o3d.io.write_point_cloud(
            '{}/test_data/ply/{}.ply'.format(CUR_DIR, mesh_filepath.split('/')[-1].split('.')[0]),
            pc)
    return v_sampled, n_sampled


def sample_mesh_surface(mesh: Union[trimesh.Trimesh, trimesh.Scene], voxel_size=0.006):
    # points, face_index = trimesh.sample.sample_surface(mesh, 50000)
    # normals = mesh.face_normals[face_index]
    # v = np.array(points).astype(np.float32)
    # n = np.array(normals).astype(np.float32)
    point_cloud = get_surface_point_cloud(mesh, scan_count=25, scan_resolution=150, sample_point_count=50000)
    v = point_cloud.points.astype(np.float32)
    n = point_cloud.normals.astype(np.float32)

    # Downsample a point cloud on a voxel grid so there is at most one point per voxel.
    # Any arguments after the points are treated as attribute arrays and get averaged within each voxel
    v_sampled = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, v)
    v_sampled = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, v_sampled)

    kdtree = KDTree(data=v)
    dist, v_indices = kdtree.query(v_sampled, k=1)
    v_sampled = v[v_indices]
    n_sampled = n[v_indices]
    return v_sampled, n_sampled


def create_table_points(lx, ly, lz, dx=0, dy=0, dz=0, grid_size=0.01):
    '''
    **Input:**
    - lx:
    - ly:
    - lz:
    **Output:**
    - numpy array of the points with shape (-1, 3).
    '''
    xmap = np.linspace(0, lx, int(lx / grid_size))
    ymap = np.linspace(0, ly, int(ly / grid_size))
    zmap = np.linspace(0, lz, int(lz / grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, ymap, zmap], axis=-1)
    points = points.reshape([-1, 3])
    return points


def create_plane_points_with_normal(lx, ly, grid_size):
    points = create_table_points(lx, ly, grid_size, dx=-lx / 2, dy=-ly / 2, dz=0, grid_size=grid_size)
    normals = np.ones_like(points)
    normals[:, :2] = 0
    return np.concatenate([points, normals], axis=1)
