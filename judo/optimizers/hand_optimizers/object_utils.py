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
from mjmanip.utils import IDENTITY_POSE, mj_model_name, mj_pose_to_mat
from mjmanip.trimesh_utils import mj_get_body_trimeshes
from mjmanip.pytorch3d_utils import p3d_transform_points


@dataclass
class ObjectData:
    geom_points: dict[str, torch.Tensor]
    geom_normals: dict[str, torch.Tensor]
    geom_meshes: dict[str, trimesh.Trimesh]
    geom_tfs: dict[str, np.ndarray]  # mat 4x4
    geom_mesh_paths: Optional[dict[str, str]] = None
    scale: float = 1.0
    npoints_each_geom: int = 1000

    @classmethod
    def get_geoms_data(cls, geom_mesh_paths: dict[str, str], voxel_size=0.006, scale=1.0, vis=False,
                       watertight_process=True,
                       npoints_each_geom: int = 1000,
                       device: Union[torch.device, str] = 'cuda') -> ObjectData:
        geom_meshes = {}
        geom_points = {}
        geom_normals = {}
        geom_tfs = {}
        for geom_name, geom_mesh_path in geom_mesh_paths.items():
            geom_mesh = trimesh.load_mesh(geom_mesh_path)
            geom_meshes[geom_name] = geom_mesh
            # scale the object mesh
            geom_mesh.vertices *= scale

            # Sample mesh with [watertight_process]
            v_sampled, n_sampled = get_watertight_vertices_normals(geom_mesh, voxel_size, watertight_process,
                                                                   geom_mesh_path, npoints_each_geom, vis)
            geom_points[geom_name] = torch.tensor(v_sampled, dtype=torch.float32, device=device)
            geom_normals[geom_name] = torch.tensor(n_sampled, dtype=torch.float32, device=device)
            geom_tfs[geom_name] = np.eye(4)

        return ObjectData(geom_points=geom_points,
                          geom_normals=geom_normals,
                          geom_mesh_paths=geom_mesh_paths,
                          geom_meshes=geom_meshes,  # scaled mesh
                          geom_tfs=geom_tfs,
                          scale=scale,
                          npoints_each_geom=npoints_each_geom)

    @classmethod
    def get_mj_object_data(cls, mj_model: Union[mj.MjModel, str], mj_data: Optional[mj.MjData] = None,
                           mj_spec: Optional[mj.MjSpec] = None,
                           meshdir: Optional[str] = None,
                           body_names: Optional[list[str]] = None,
                           geom_names: Optional[list[str]] = None,
                           voxel_size: float = 0.006, scale: float = 1.0,
                           merging_meshes: bool = False,
                           npoints_each_geom: int = 1000,
                           is_collision: bool = False,
                           visualized: bool = False,
                           device: Union[torch.device, str] = 'cuda') -> ObjectData:
        if isinstance(mj_model, str):
            assert mj_model.endswith('.xml')
            mj_spec = mj.MjSpec.from_file(mj_model)
            mj_model: mj.MjModel = mj_spec.compile()
            if not mj_data:
                mj_data = mj.MjData(mj_model)
        else:
            pass

        if not body_names:
            body_names = [mj_model.body(i).name for i in range(mj_model.nbody)]
            body_names.remove('world')
        geom_meshes: dict[str, trimesh.Trimesh] = {}
        geom_mesh_paths: dict[str, str] = {}
        geom_tfs: dict[str, np.ndarray] = {}
        for geom_name, m in mj_get_body_trimeshes(mj_model, data=mj_data, model_spec=mj_spec, meshdir=meshdir,
                                                  body_names=body_names, geom_names=geom_names,
                                                  use_global_pose=True, is_collision=is_collision).items():
            geom_meshes[geom_name] = m[0]
            geom_mesh_paths[geom_name] = m[1]
            geom_tfs[geom_name] = m[2]
        if merging_meshes:
            model_name = mj_model_name(mj_model)
            geom_meshes = {model_name: trimesh.util.concatenate(geom_meshes.values())}
            base_body = mj_model.body(body_names[0])
            base_body_data = mj_data.body(body_names[0]) if mj_data else None
            geom_tfs[model_name] = mj_pose_to_mat(np.concat([base_body_data.xpos, base_body_data.xquat]) \
                                                      if base_body_data else np.concat([base_body.pos, base_body.quat]))

        # Sample meshes' points
        geom_points = {}
        geom_normals = {}
        for geom_name, geom_mesh in geom_meshes.items():
            # scale the object mesh
            geom_mesh.vertices *= scale

            # Sample mesh with [watertight_process]
            v_sampled, n_sampled = get_watertight_vertices_normals(geom_mesh, voxel_size, watertight_process=True,
                                                                   nsample_points=npoints_each_geom)
            geom_points[geom_name] = torch.tensor(v_sampled, dtype=torch.float32, device=device)
            geom_normals[geom_name] = torch.tensor(n_sampled, dtype=torch.float32, device=device)

        if visualized:
            trimesh.Scene(geom_meshes).show()
            trimesh.Scene(trimesh.PointCloud(torch.cat(list(geom_points.values())).detach().cpu().numpy())).show()
        return ObjectData(geom_points=geom_points,
                          geom_normals=geom_normals,
                          geom_mesh_paths=geom_mesh_paths,
                          geom_meshes=geom_meshes,  # already scaled meshes
                          geom_tfs=geom_tfs,
                          scale=scale)

    @property
    def all_points(self) -> torch.Tensor:
        return torch.cat(list(self.geom_points.values()))

    @property
    def all_normals(self) -> torch.Tensor:
        return torch.cat(list(self.geom_normals.values()))

    def transform_to(self, new_geom_poses: dict[str, np.ndarray], resample: bool = False):
        assert new_geom_poses.keys() == self.geom_meshes.keys()
        new_geom_tfs = {}
        for geom_name, geom_mesh in self.geom_meshes.items():
            device = self.geom_points[geom_name].device
            # print("ObjectData.transform_to", geom_name, self.geom_mesh_paths[geom_name])

            # Cur geom pose
            cur_geom_tf = self.geom_tfs[geom_name]

            # New geom pose
            new_geom_pose = new_geom_poses[geom_name]
            new_geom_tf = mj_pose_to_mat(new_geom_pose)
            new_geom_tfs[geom_name] = new_geom_tf

            # Apply [new_pose] to mesh in place
            delta_tf = new_geom_tf @ np.linalg.inv(cur_geom_tf)

            # Resample the mesh
            # NOTE: points & normals dtype are torch.float32 or float32 for convenient operations later
            if resample:
                geom_mesh.apply_transform(delta_tf)
                v_sampled, n_sampled = sample_mesh_surface(geom_mesh)
                # NOTE: Points, normals can have new lengths so not copying here
                self.geom_points[geom_name] = torch.tensor(v_sampled, dtype=torch.float32, device=device)
                self.geom_normals[geom_name] = torch.tensor(n_sampled, dtype=torch.float32, device=device)
            else:
                delta_tf_torch = torch.tensor(delta_tf, dtype=torch.float32, device=device)
                R = delta_tf_torch[:3, :3]
                t = delta_tf_torch[:3, 3]
                self.geom_points[geom_name] = self.geom_points[geom_name] @ R.T + t
                self.geom_normals[geom_name] = torch.nn.functional.normalize(self.geom_normals[geom_name] @ R.T, dim=-1)
        self.geom_tfs = new_geom_tfs

    def visualize(self):
        trimesh.Scene(self.geom_meshes).show()
        trimesh.Scene(trimesh.PointCloud(self.all_points.cpu().numpy())).show()


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
                                    nsample_points: int = 1000,
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

    v_sampled, n_sampled = sample_mesh_surface(mesh, voxel_size, nsample_points)
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


def sample_mesh_surface(mesh: Union[trimesh.Trimesh, trimesh.Scene], voxel_size=0.006, nsample_points: int = 50000):
    # points, face_index = trimesh.sample.sample_surface(mesh, 50000)
    # normals = mesh.face_normals[face_index]
    # v = np.array(points).astype(np.float32)
    # n = np.array(normals).astype(np.float32)
    point_cloud = get_surface_point_cloud(mesh, scan_count=25, scan_resolution=150, sample_point_count=nsample_points)
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
