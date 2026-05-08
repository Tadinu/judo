# leap_hand layer for torch
from typing import Optional, Union, TYPE_CHECKING
import torch
import trimesh
import os
import numpy as np
import copy
from itertools import chain

from pathlib import Path

import coacd
from mesh_to_sdf import get_surface_point_cloud
from scipy.spatial import KDTree
import point_cloud_utils as pcu
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from mjmanip.pytorch3d_utils import p3d_transform_points

import mujoco as mj
import roma

from .layer_asset_utils import LEAP_LAYER_CACHE_DIR

if TYPE_CHECKING:
    from ..optimizers.hand_optimizers.hand_optimizer import HandParams

# mjmanip
from mjmanip.utils import mj_get_mesh_file_path
from mjmanip.trimesh_utils import mj_geom_spec_to_trimesh

USE_LEAP_MJX = True
if USE_LEAP_MJX:
    from mjmanip.robot.leap_mjx import LEAP_ASSETS_DIR, LeapMjx

    LEAP = LeapMjx
else:
    from mjmanip.robot.leap import LEAP_ASSETS_DIR, Leap

    LEAP = Leap


# All lengths are in mm and rotations in radians


class LeapHandLayer(torch.nn.Module):

    def float_torch_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).to(device=self.device, dtype=torch.float32).repeat(self.nbatches, 1)

    def __init__(self, hand_model_desc: str,
                 hand_base_pose: Optional[np.ndarray] = None,
                 joint_angles: Optional[np.ndarray] = None,
                 batch_size: int = 1,
                 to_mano_frame: bool = False, show_mesh: bool = False,
                 use_collision_mesh: bool = False, regen_cache: bool = False,
                 visualized: bool = False, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.nbatches = batch_size
        self.hand_params: Optional[HandParams] = None
        self.hand_model_desc = hand_model_desc
        self.mj_init_hand_base_pose = np.tile(np.array([0, 0, 0, 1, 0, 0, 0]), (batch_size, 1))
        torch_hand_base_pose = torch.eye(4).reshape(-1, 4, 4).float()
        if hand_base_pose is not None:
            assert hand_base_pose.shape[0] == batch_size
            self.mj_init_hand_base_pose = hand_base_pose
            torch_hand_base_pose[:, :3, 3] = torch.from_numpy(hand_base_pose[:, 3]).float().to(device)
            torch_hand_base_pose[:, :3, :3] = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(
                torch.from_numpy(hand_base_pose[:, 3:]).float().to(device)))
        self.hand_base_pose = torch_hand_base_pose

        self.show_mesh = show_mesh
        self.make_contact_points = True
        self.use_collision_mesh = use_collision_mesh
        self.to_mano_frame = to_mano_frame
        self.visualized = visualized
        self.finger_num = 4
        self.mj_init_joint_anges = joint_angles if joint_angles is not None \
            else np.tile(np.zeros(LEAP.HAND_DOFS_NO), (batch_size, 1))
        self.joint_angles = torch.from_numpy(joint_angles).float().to(device) if joint_angles is not None else None
        self.batch_size = batch_size

        self.regen_cache = regen_cache
        self.geom_trimeshes: dict[str, trimesh.Trimesh] = {}
        self.geom_convex_meshes: dict[str, trimesh.Trimesh] = \
            self.load_assets(f"{LEAP_LAYER_CACHE_DIR}/hand_meshes_cvx", is_mesh=True)
        self.hand_surface_points: dict[str, np.ndarray] = self.load_assets(f"{LEAP_LAYER_CACHE_DIR}/hand_points")
        self.visible_point_indices: dict[str, np.ndarray] = (
            self.load_assets(f"{LEAP_LAYER_CACHE_DIR}/visible_point_indices"))
        self.hand_composite_points: dict[str, np.ndarray] = {}
        # NOTE: These are from 'leap_rh_mjx.xml'
        self.link_geom_names = {
            # palm
            'leap_mount': ['leap_mount_collision_0', 'leap_mount_collision_1'],
            'palm': [f'palm_collision_{i}' for i in range(1, 11)],
            # thumb
            'th_mp': 'th_mp_collision',
            'th_px': 'th_px_collision',
            'th_ds': ['th_ds_collision', 'th_tip_collision'],
            # index
            'if_bs': 'if_bs_collision', 'if_px': 'if_px_collision',
            'if_md': ['if_md_collision_1', 'if_md_collision_2'],
            'if_ds': ['if_ds_collision', 'if_tip_collision'],
            # middle
            'mf_bs': 'mf_bs_collision', 'mf_px': 'mf_px_collision',
            'mf_md': ['mf_md_collision_1', 'mf_md_collision_2'],
            'mf_ds': ['mf_ds_collision', 'mf_tip_collision'],
            # ring
            'rf_bs': 'rf_bs_collision', 'rf_px': 'rf_px_collision',
            'rf_md': ['rf_md_collision_1', 'rf_md_collision_2'],
            'rf_ds': ['rf_ds_collision', 'rf_tip_collision'],
        } if self.use_collision_mesh else {
            # palm
            'leap_mount': 'leap_mount_visual', 'palm': 'palm_visual',
            # thumb
            'th_mp': 'th_mp_visual', 'th_bs': 'th_bs_visual',
            'th_px': 'th_px_visual',
            'th_ds': ['th_ds_visual', 'th_tip_visual'],
            # index
            'if_bs': 'if_bs_visual', 'if_px': 'if_px_visual', 'if_md': 'if_md_visual',
            'if_ds': ['if_ds_visual', 'if_tip_visual'],
            # middle
            'mf_bs': 'mf_bs_visual', 'mf_px': 'mf_px_visual', 'mf_md': 'mf_md_visual',
            'mf_ds': ['mf_ds_visual', 'mf_tip_visual'],
            # ring
            'rf_bs': 'rf_bs_visual', 'rf_px': 'rf_px_visual', 'rf_md': 'rf_md_visual',
            'rf_ds': ['rf_ds_visual', 'rf_tip_visual'],
        }
        self.ordered_finger_endeffort = list(chain(*[self.link_geom_names[link] for link in
                                                     ['leap_mount', 'palm',
                                                      'th_ds', 'if_ds', 'mf_ds', 'rf_ds']]))

        # transformation for align the robot hand to mano hand frame, used for
        self.to_mano_transform = torch.eye(4).float().to(device)
        if self.to_mano_frame:
            self.to_mano_transform[:3, :] = torch.tensor([[-1, 0, 0, 0],
                                                          [0, 0, 1, 0.0175],
                                                          [0, 1, 0, 0.0375]])

        self.register_buffer('base_2_world', self.to_mano_transform)

    def init_kinematics(self):
        self.is_from_urdf = self.hand_model_desc.endswith('urdf')
        if self.is_from_urdf:
            self.chain = pk.build_chain_from_urdf(open(self.hand_model_desc).read()).to(device=self.device)
        else:
            self.mj_spec, self.chain = pk.build_chain_from_mjcf(self.hand_model_desc, device=self.device)
            self.mj_spec.meshdir = LEAP_ASSETS_DIR
            self.mj_model: mj.MjModel = self.mj_spec.compile()
            self.mj_data: mj.MjData = mj.MjData(self.mj_model)
            self.hand_base = self.mj_model.body(LEAP.HAND_BASE_NAME)
            self.hand_body_names = [self.mj_model.body(i).name for i in range(self.mj_model.nbody)]
            self.hand_body_names.remove('world')

        self.joint_lowers = self.chain.low
        self.joint_uppers = self.chain.high
        self.joint_means = (self.joint_lowers + self.joint_uppers) / 2
        self.joint_ranges = self.joint_means - self.joint_lowers
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_dofs = self.chain.n_joints  # only used here for robot hand with no mimic joint

        # Create cache data
        if self.regen_cache or not (self.geom_convex_meshes and
                                    self.hand_surface_points and self.visible_point_indices):
            self.create_assets()
            self.regen_cache = True
        else:
            self.make_contact_points = False
            self.meshes_data = self.load_meshes()

        self.hand_segment_indices, self.hand_finger_indices = self.get_hand_segment_indices()

    @classmethod
    def load_assets(cls, dir_path: str, is_mesh: bool = False) -> dict[str, Union[trimesh.Trimesh, np.ndarray]]:
        load_data = trimesh.load if is_mesh else np.load
        return {Path(filename).stem: load_data(os.path.join(root, filename))
                for root, dirs, files in os.walk(dir_path)
                for filename in files}

    def create_assets(self):
        '''
        To create needed assets for the first running.
        Should run before first use.
        '''
        pose = self.hand_base_pose
        hand_qpos = self.joint_angles

        show_mesh = self.show_mesh
        self.show_mesh = True  # mesh with face
        self.make_contact_points = True  # True: creating convex meshes

        self.meshes_data = self.load_meshes()
        self.save_geom_convex_meshes(self.geom_convex_meshes)

        hand_meshes = self.get_forward_hand_mesh(pose, hand_qpos)[0]
        hand_parts = hand_meshes.split()

        hand_single_mesh = trimesh.boolean.boolean_manifold(hand_parts, 'union') if self.is_from_urdf else \
            trimesh.util.concatenate(hand_parts)
        hand_single_mesh.export(f'{LEAP_LAYER_CACHE_DIR}/hand.obj')

        self.show_mesh = True
        self.make_contact_points = False
        self.meshes_data = self.load_meshes()
        hand_all_zero_mesh = self.get_forward_hand_mesh(pose, hand_qpos)[0]
        hand_all_zero_mesh.export(f'{LEAP_LAYER_CACHE_DIR}/hand_all_zero.obj')

        self.show_mesh = False
        self.make_contact_points = True
        self.meshes_data = self.load_meshes()
        self.save_surface_points(self.hand_surface_points)

        self.get_forward_vertices(pose, hand_qpos)  # Sample [hand_composite_points]
        self.visible_point_indices = self.sample_visible_points(hand_single_mesh, self.hand_composite_points,
                                                                down_sampling=False)
        self.show_mesh = True
        self.make_contact_points = False
        self.meshes_data = self.load_meshes()
        hand_to_mano_mesh = self.get_forward_hand_mesh(pose, hand_qpos)[0]
        hand_to_mano_mesh.export(f'{LEAP_LAYER_CACHE_DIR}/hand_to_mano_frame.obj')

        self.make_contact_points = False
        self.show_mesh = show_mesh

    def load_meshes(self) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        meshes_data = {}
        for link_key, _ in self.link_geom_names.items():
            link = self.chain.find_link(link_key)
            if not link:
                continue
            link_pk_geoms = link.collisions if self.use_collision_mesh else link.visuals
            for pk_geom in link_pk_geoms:
                geom_spec = next((g for g in self.mj_spec.geoms if g.name == pk_geom.name), None)
                if geom_spec is None:
                    continue

                # 1- Load [geom_mesh], saving it to [self.geom_meshes]
                geom_mesh = mj_geom_spec_to_trimesh(self.mj_spec, geom_spec)
                if not geom_mesh:
                    continue
                geom_name = geom_spec.name
                self.geom_trimeshes[geom_name] = geom_mesh

                # 2- Turn [geom_mesh] to convex mesh, saving it to [self.geom_convex_meshes]
                if self.make_contact_points:
                    threshold = 0.3
                    use_convex_hull = True or 'palm' in geom_name
                    if use_convex_hull:
                        geom_mesh = geom_mesh.convex_hull
                    else:
                        coacd_mesh = coacd.Mesh(geom_mesh.vertices, geom_mesh.faces)
                        parts = coacd.run_coacd(coacd_mesh, threshold=threshold, apx_mode='ch')
                        coacd_meshes = []
                        for part in parts:
                            coacd_meshes.append(trimesh.Trimesh(vertices=part[0], faces=part[1]))
                        geom_mesh = trimesh.boolean.boolean_manifold(coacd_meshes, 'union')
                    self.geom_convex_meshes[geom_name] = geom_mesh

                geom_pre_transform = pk_geom.offset
                # print(geom_name, link.offset, geom_pre_transform)
                if self.show_mesh:
                    geom_verts = torch.from_numpy(geom_mesh.vertices).float().to(self.device)
                    geom_vertex_normals = (torch.from_numpy(copy.deepcopy(geom_mesh.vertex_normals)).float()
                                           .to(self.device))
                    geom_verts = geom_pre_transform.transform_points(geom_verts)
                    geom_vertex_normals = geom_pre_transform.transform_normals(geom_vertex_normals)

                    temp = torch.ones(geom_mesh.vertices.shape[0], 1).float()
                    meshes_data[geom_name] = [
                        torch.cat((geom_verts, temp), dim=-1).to(self.device),
                        geom_mesh.faces,
                        torch.cat((geom_vertex_normals, temp), dim=-1).float().to(self.device)
                    ]
                else:
                    points, point_normals = self.sample_geom_surface_points(geom_mesh)
                    self.hand_surface_points[geom_name] = np.concatenate([points, point_normals], axis=-1)
                    points_info = self.hand_surface_points[geom_name]
                    if self.make_contact_points:
                        idxs = np.arange(len(points_info))
                    else:
                        idxs = self.visible_point_indices[geom_name]

                    geom_verts = torch.from_numpy(points_info[idxs, :3]).float().to(self.device)
                    geom_verts = geom_pre_transform.transform_points(geom_verts)

                    geom_vertex_normals = torch.from_numpy(points_info[idxs, 3:6]).float().to(self.device)
                    geom_vertex_normals = geom_pre_transform.transform_normals(geom_vertex_normals)
                    temp = torch.ones(idxs.shape[0], 1).float()
                    meshes_data[geom_name] = [
                        torch.cat((geom_verts, temp), dim=-1).to(self.device),
                        torch.zeros([0]),  # no real meaning, just for placeholder
                        torch.cat((geom_vertex_normals, temp), dim=-1).float().to(self.device)
                    ]
        return meshes_data

    def get_hand_segment_indices(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        hand_segment_indices = {}
        hand_finger_indices = {}
        segment_start = 0  # torch.tensor(0, dtype=torch.long, device=self.device)
        finger_start = 0  # torch.tensor(0, dtype=torch.long, device=self.device)
        for geom_name, geom_mesh_meta in self.meshes_data.items():
            geom_name = Path(geom_name).stem
            end = segment_start + geom_mesh_meta[0].shape[0]
            hand_segment_indices[geom_name] = torch.arange(segment_start, end)  # [segment_start, end]
            if geom_name in self.ordered_finger_endeffort:
                hand_finger_indices[geom_name] = torch.arange(finger_start, end)  # [finger_start, end]
                finger_start = end
            segment_start = end
        return hand_segment_indices, hand_finger_indices

    def forward(self, hand_qpos: torch.Tensor):
        """
        Args:
            hand_qpos (Tensor (batch_size x 15)): The degrees of freedom of the Robot hand.
       """
        return self.chain.forward_kinematics(hand_qpos)

    def compute_abnormal_joint_loss(self, hand_qpos: torch.Tensor):
        loss_1 = -torch.clamp(hand_qpos[:, 5] - hand_qpos[:, 1], -1, 0) * 10
        loss_2 = -torch.clamp(hand_qpos[:, 9] - hand_qpos[:, 5], -1, 0) * 10
        loss_3 = (torch.abs(hand_qpos[:, [2, 3, 6, 7, 10, 11]] - self.joint_means[[2, 3, 6, 7, 10, 11]].unsqueeze(0))
                  .sum(dim=-1) * 2)
        return loss_1 + loss_2 + loss_3

    def get_init_angle(self):
        init_angle = (self.joint_uppers - self.joint_lowers) / 6.0 + self.joint_lowers
        init_angle[1] = -0.1
        init_angle[5] = 0.0
        init_angle[9] = 0.1
        init_angle[12] = 0.8
        return init_angle

    def get_hand_mesh(self, hand_base_pose, frame_tfs: dict[str, tf.Transform3d]) -> trimesh.Trimesh:
        bs = hand_base_pose.shape[0]

        meshes = []
        for link_key, geom_name_list in self.link_geom_names.items():
            geom_name_list = [geom_name_list] if isinstance(geom_name_list, str) else geom_name_list
            for geom_name in geom_name_list:
                mesh_data = self.meshes_data[geom_name]
                rotmat = torch.matmul(hand_base_pose,
                                      torch.matmul(self.to_mano_transform, frame_tfs[link_key].get_matrix()))
                vertices = mesh_data[0]
                batch_vertices = p3d_transform_points(vertices, rotmat)
                face = mesh_data[1]
                sub_meshes = [trimesh.Trimesh(vertices.cpu().numpy(), face) for vertices in batch_vertices]
                meshes.append(sub_meshes)

        hand_meshes = []
        for j in range(bs):
            hand_mesh = np.sum([meshes[i][j] for i in range(len(meshes))])
            hand_meshes.append(hand_mesh)
        # trimesh.Scene(hand_meshes).show()
        return hand_meshes

    def get_forward_hand_mesh(self, hand_base_pose: torch.Tensor, hand_qpos: torch.Tensor) -> trimesh.Trimesh:
        frame_tfs = self.forward(hand_qpos)
        return self.get_hand_mesh(hand_base_pose, frame_tfs)

    def get_forward_vertices(self, hand_base_pose: torch.Tensor, hand_qpos: torch.Tensor):
        """
        NOTE: This function must be differentiable, so purely written in Torch!
        """
        frame_tfs = self.forward(hand_qpos)

        verts = []
        verts_normal = []

        # for mesh_key, mesh in self.meshes_meta.items():
        for link_key, geom_names in self.link_geom_names.items():
            geom_name_list = [geom_names] if isinstance(geom_names, str) else geom_names
            for geom_name in geom_name_list:
                mesh_data = self.meshes_data.get(geom_name, None)
                if mesh_data is None:
                    print(f"mesh {geom_name} not found")
                    continue

                # hand_base_pose * frame_pose
                frame_pose = frame_tfs[link_key].get_matrix()
                link_pose = torch.matmul(hand_base_pose, torch.matmul(self.to_mano_transform,
                                                                      frame_pose) if self.to_mano_frame else frame_pose)

                # hand_base_pose * link_pose * link_geom_verts/normals
                vertices = mesh_data[0]
                batch_vertices = p3d_transform_points(vertices, link_pose)
                verts.append(batch_vertices)
                if self.make_contact_points:
                    self.hand_composite_points[geom_name] = batch_vertices.squeeze().detach().cpu().numpy()

                vertex_normals = mesh_data[2]
                link_pose[:, :3, 3] *= 0
                batch_vertex_normals = p3d_transform_points(vertex_normals, link_pose)
                verts_normal.append(batch_vertex_normals)

        verts = torch.cat(verts, dim=1).contiguous()
        verts_normal = torch.cat(verts_normal, dim=1).contiguous()
        return verts, verts_normal

    @classmethod
    def save_geom_convex_meshes(cls, geom_convex_meshes: dict[str, trimesh.Trimesh],
                                dst_dir=f'{LEAP_LAYER_CACHE_DIR}/hand_meshes_cvx'):
        os.makedirs(dst_dir, exist_ok=True)
        for geom_name, convex_mesh in geom_convex_meshes.items():
            filepath = f"{dst_dir}/{geom_name}.stl"
            convex_mesh.export(filepath)
            # print('saved:', filepath)
        print('saving convex meshes done', dst_dir)

    @classmethod
    def sample_visible_points(cls, hand_whole_mesh: trimesh.Trimesh,
                              hand_composite_points: dict[str, np.ndarray],
                              voxel_size=0.0055, dist_threshold: float = 0.0005,
                              down_sampling: bool = False) -> dict[str, np.ndarray]:
        count = 0
        dst_dir = f'{LEAP_LAYER_CACHE_DIR}/visible_point_indices'
        os.makedirs(dst_dir, exist_ok=True)

        # hand = trimesh.load(f'{LEAP_LAYER_CACHE_DIR}/hand.obj', force='mesh')
        hand_surface_pcl = get_surface_point_cloud(hand_whole_mesh, scan_count=100, scan_resolution=200)
        hand_surface_points = hand_surface_pcl.points.view(np.ndarray)
        if down_sampling:
            hand_surface_points = pcu.downsample_point_cloud_on_voxel_grid(voxel_size / 2, hand_surface_points)
            hand_surface_points = pcu.downsample_point_cloud_on_voxel_grid(voxel_size / 2, hand_surface_points)

        point_tree = KDTree(data=hand_surface_points)
        visible_point_indices = {}
        for geom_name, geom_points in hand_composite_points.items():
            dist, index = point_tree.query(geom_points[:, :3], k=1)
            mask = dist < dist_threshold
            index = index[mask]

            visible_points = geom_points[mask]
            v_sampled = visible_points[:, :3]
            if down_sampling:
                v_sampled = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, v_sampled)
                if v_sampled.any():
                    v_sampled = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, v_sampled)
            if not v_sampled.any():
                continue

            # v_sampled = o3d_vox_downsample(visible_points, 0.005)
            count += len(v_sampled)
            # pc = trimesh.PointCloud(v_sampled, colors=(255, 255, 0))
            # pc.show()

            _, v_sampled_idx = KDTree(geom_points[:, :3]).query(v_sampled, k=1)
            # pc = trimesh.PointCloud(point_info[:, :3][v_sampled_idx])
            # pc.show()
            hand_surface_points[index] = np.array([-1e6, -1e6, 1e6])
            np.save(f'{dst_dir}/{geom_name}.npy', v_sampled_idx)
            visible_point_indices[geom_name] = v_sampled_idx
        print("sampling visible points done:", count, dst_dir)
        return visible_point_indices

    @classmethod
    def sample_geom_surface_points(cls, geom_mesh: trimesh.Trimesh, npoints: int = 50000) \
            -> tuple[np.ndarray, np.ndarray]:
        np.random.seed(0)
        points, idx = trimesh.sample.sample_surface_even(geom_mesh, npoints, radius=None)
        point_normals = geom_mesh.face_normals[idx]
        return points.view(np.ndarray), point_normals

    @classmethod
    def save_surface_points(cls, surface_points: dict[str, np.ndarray],
                            dst_dir=f'{LEAP_LAYER_CACHE_DIR}/hand_points',
                            visualize: bool = False):
        os.makedirs(dst_dir, exist_ok=True)
        for geom_name, points_info in surface_points.items():
            np.save(os.path.join(dst_dir, f"{geom_name}.npy"), points_info)
            # print(f"saved {geom_name} surface points")
            if visualize:
                points = points_info[:, :3]
                point_normals = points_info[:, 3:6]
                pc = trimesh.PointCloud(points, colors=(255, 255, 0))
                ray_visualization = trimesh.load_path(np.hstack((points,
                                                                 points + point_normals / 100)).reshape(-1, 2, 3))
                trimesh.Scene([pc, ray_visualization]).show()
        print("saving surface points done", dst_dir)


class LeapAnchor(torch.nn.Module):
    # Default anchor points
    DEFAULT_VERT_IDXES = np.array(
        [647, 647, 690, 690, 461, 461, 377, 377, 386, 386, 184, 184, 146, 146, 1152, 1152, 1151, 1023, 1023, 888, 888,
         828, 828, 839, 839, 179, 179, 1561, 1625, 1625, 1529, 1529, 1350, 1424, 1424, 1304, 1304, 1331, 164, 164, 205,
         205, 186, 186, 2074, 2017, 2017, 2065, 2065, 1953, 1953, 1993, 1993, 1782, 1782, 1697, 1697])

    def __init__(self):
        super().__init__()
        self.picking_points = not len(self.DEFAULT_VERT_IDXES)
        self.register_buffer("vert_idx", torch.from_numpy(self.DEFAULT_VERT_IDXES).long())

    def forward(self, vertices: torch.Tensor):
        """
        vertices: TENSOR[N_BATCH, 4040, 3]
        """
        if self.picking_points:
            vert_idx = self.pick_points(vertices.squeeze().detach().cpu().numpy())
            self.register_buffer("vert_idx", torch.from_numpy(vert_idx).long())
            self.picking_points = False

        assert vertices.shape[1] > self.vert_idx.max(), \
            f"You may wanna increase n_points in sample_geom_surface_points()"
        anchor_pos = vertices[:, self.vert_idx, :]
        return anchor_pos

    @classmethod
    def pick_points(cls, vertices: np.ndarray) -> np.ndarray:
        import open3d as o3d
        print("1) Please pick at least three correspondences using [shift + left click]")
        print("Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print(vis.get_picked_points())
        return np.array(vis.get_picked_points())


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    show_mesh = False
    to_mano_frame = True
    hand = LeapHandLayer(show_mesh=show_mesh, to_mano_frame=to_mano_frame, device=device)

    hand_base_pose = torch.eye(4).to(device).reshape(-1, 4, 4).float().to(device)
    hand_qpos = np.zeros((1, 16), dtype=np.float32)
    hand_qpos[0, :4] = np.array([0.0, 0.0, 0, 0])
    hand_qpos = torch.from_numpy(hand_qpos).to(device)

    # mesh version
    if show_mesh:
        hand_mesh = hand.get_forward_hand_mesh(hand_base_pose, hand_qpos)[0]
        hand_mesh.show()
    else:
        hand_verts, hand_normals = hand.get_forward_vertices(hand_base_pose, hand_qpos)
        pc = trimesh.PointCloud(hand_verts.squeeze().cpu().numpy(), colors=(0, 255, 255))

        hand_data = np.hstack((hand_verts[0].detach().cpu().numpy(),
                               hand_verts[0].detach().cpu().numpy() +
                               hand_normals[0].detach().cpu().numpy() * 0.01)).reshape(-1, 2, 3)
        ray_visualize = trimesh.load_path(hand_data)
        scene = trimesh.Scene([pc, ray_visualize])
        scene.show()

        hand_mano_mesh = trimesh.load(f'{LEAP_LAYER_CACHE_DIR}/hand_to_mano_frame.obj')
        anchor_layer = LeapAnchor()
        anchors = anchor_layer(hand_verts).squeeze().cpu().numpy()
        pc_anchors = trimesh.PointCloud(anchors, colors=(0, 0, 255))
        scene = trimesh.Scene([hand_mano_mesh, pc, pc_anchors, ray_visualize])
        scene.show()
