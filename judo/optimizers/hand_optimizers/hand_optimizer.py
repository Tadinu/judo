# Ref: https://github.com/DexGraspOpt/DexGraspSyn
from dataclasses import dataclass
from typing import Any, Optional, Union
import numpy as np

# torch
import torch
from torch.optim.lr_scheduler import StepLR as TorchStepLR

# pytorch3d
# https://miropsota.github.io/torch_packages_builder/pytorch3d (from versions of torch + cuda + python)
import pytorch3d
import pytorch3d.structures
import pytorch3d.ops

import roma
import transforms3d
import trimesh
from meshlib import mrmeshpy
from meshlib import mrmeshnumpy as mrn
from tqdm import tqdm

# judo
from judo.optimizers.hand_optimizers.object_utils import ObjectData
from judo.optimizers.hand_optimizers.loss_utils import point2point_signed
from judo.optimizers.hand_optimizers.rot6d import (robust_compute_rotation_matrix_from_ortho6d,
                                                   compute_rotation_ortho6d_from_matrix)

USE_MUJOCO_HAND_LAYER = True
USE_NEWTON_HAND_LAYER = False and not USE_MUJOCO_HAND_LAYER
if USE_MUJOCO_HAND_LAYER:
    from judo.hand_layers.leap_layer_mujoco import MJLeapHandLayer as LeapHandLayer, USE_MJ_WARP
elif USE_NEWTON_HAND_LAYER:
    from judo.hand_layers.leap_layer_newton import NTLeapHandLayer as LeapHandLayer
else:
    from judo.hand_layers.leap_layer import LeapHandLayer
from judo.hand_layers.leap_layer import LeapAnchor

# mjmanip
from mjmanip.trimesh_utils import mj_get_body_trimeshes
from mjmanip.pytorch3d_utils import mjw_geoms_to_pytorch3d_meshes, p3d_to_trimesh
from mjmanip.o3d_utils import o3d_vox_downsample


# roma quat [x, y, z, w]
@dataclass
class HandOptimizerParams:
    nbatches: int = 1
    distance_lower: float = 0.05
    distance_upper: float = 0.15
    jitter_strength: float = 0.1
    joint_limit_lower: float = -np.pi / 6
    joint_limit_upper: float = np.pi / 6


@dataclass
class HandParams:
    hand_model_name: str
    joint_angles: Optional[torch.Tensor] = None  # (1, ndofs)
    wrist_pos: Optional[torch.Tensor] = None  # (1, 3)
    wrist_rot6d: Optional[torch.Tensor] = None  # (1, 6)
    wrist_quat_wxyz: Optional[torch.Tensor] = None  # (1, 4)
    parallel_contact_point: Optional[torch.Tensor] = None
    xml_path: Optional[str] = None
    urdf_path: Optional[str] = None

    @classmethod
    def get(cls, hand_model_name: str, xml_path: str,
            joint_angles: np.ndarray[np.float32],
            hand_pos: np.ndarray[np.float32], hand_quat: np.ndarray[np.float32], device: str = 'cuda'):
        wrist_quat = torch.from_numpy(hand_quat).to(device).unsqueeze(0)
        return HandParams(hand_model_name=hand_model_name,
                          xml_path=xml_path,
                          joint_angles=torch.from_numpy(joint_angles).to(device).unsqueeze(0),
                          wrist_pos=torch.from_numpy(hand_pos).to(device).unsqueeze(0),
                          wrist_quat_wxyz=wrist_quat,
                          wrist_rot6d=compute_rotation_ortho6d_from_matrix(
                              roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(wrist_quat))))

    @property
    def wrist_pose(self) -> torch.Tensor:
        return torch.cat([self.wrist_pos, self.wrist_quat_wxyz], dim=-1).squeeze()


@dataclass
class HandGrasp:
    wrist_quat: Optional[np.ndarray] = None
    wrist_pos: Optional[np.ndarray] = None
    joint_angles: Optional[np.ndarray] = None
    obj_scale: float = 1.0
    obj_mesh_paths: Optional[list[str]] = None

    @property
    def wrist_pose(self):
        return np.concatenate([self.wrist_pos, self.wrist_quat], axis=-1)


class HandOptimizer(torch.nn.Module):
    """Custom Pytorch model for gradient-base grasp optimization.
    """

    def __init__(self, hand_params: HandParams,
                 object_data: Optional[ObjectData] = None,
                 obstacle_data: Optional[ObjectData] = None,
                 to_mano_frame: bool = False,
                 opt_params: Optional[HandOptimizerParams] = None,
                 apply_force_closure: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.opt_params = opt_params
        self.nbatches = opt_params.nbatches if opt_params else 1
        # Note: Force closure loss does not play much help in our observation!!!
        self.apply_force_closure = apply_force_closure

        # Object/Obstacle data
        assert object_data
        self.object_data = object_data
        self.obstacle_data = obstacle_data

        # Hand data
        self.use_quat = True  # rot6d yields a bit better grasp pose result than quat
        self.hand_params = hand_params
        self.hand_model_name = hand_params.hand_model_name
        self.hand_anchors: torch.Tensor = None
        self.hand_anchors_normals: torch.Tensor = None
        self.parallel_contact_points = hand_params.parallel_contact_point

        # 1- Init hand layer
        self.hand_layer, self.hand_anchor_layer = self._init_hand_layer(hand_params.wrist_pose.cpu().numpy(),
                                                                        hand_params.joint_angles.cpu().numpy(),
                                                                        to_mano_frame)
        self.use_mano_frame = self.hand_layer.to_mano_frame
        self.hand_visual_verts: np.ndarray = None
        self.hand_visual_vert_normals: np.ndarray = None
        self.hand_visual_grasp_site_pos: np.ndarray = None

        # 2- Grasp space (obj + hand_params' wrist_pose, joint_angles)
        self._initialize_grasp_space(self.hand_layer, hand_params)

        # 3- Hand optimizing configs
        self._init_hand_opt(hand_params)

        # Grasp site
        grasp_site = self.hand_layer.mj_model.site("grasp_site")
        self.grasp_site_local_pos = (
            torch.cat([torch.from_numpy(grasp_site.pos).repeat(self.nbatches, 1).float().to(device=self.device),
                       torch.ones((self.nbatches, 1), device=self.device)], dim=-1).float())

    def _init_hand_layer(self, hand_base_pose: np.ndarray, joint_angles: np.ndarray, to_mano_frame: bool = False) \
            -> Optional[tuple[LeapHandLayer, LeapAnchor]]:
        if "leap" in self.hand_model_name:
            hand_model_desc = self.hand_params.xml_path or self.hand_params.urdf_path
            hand_layer = LeapHandLayer(hand_model_desc=hand_model_desc,
                                       hand_base_pose=hand_base_pose,
                                       joint_angles=joint_angles,
                                       to_mano_frame=to_mano_frame,
                                       use_collision_mesh=False,
                                       regen_cache=False,
                                       visualized=True,
                                       device=self.device)
            return hand_layer, LeapAnchor()
        return None

    def _init_hand_opt(self, hand_params: HandParams):
        assert self.nbatches == hand_params.joint_angles.shape[0]
        self.hand_ndofs = hand_params.joint_angles.shape[1]
        self.joint_means = self.hand_layer.joint_means[-self.hand_ndofs:].to(self.device)
        self.joint_ranges = self.hand_layer.joint_ranges[-self.hand_ndofs:].to(self.device)
        self.finger_indices = self.hand_layer.hand_finger_indices
        if "leap" in self.hand_model_name or "allegro" in self.hand_model_name:
            self.fingers_num = 4
        elif "shadow_hand" in self.hand_model_name or "svh_hand" in self.hand_model_name:
            self.fingers_num = 5
        elif "mano_hand" in self.hand_model_name:
            self.fingers_num = 5
        else:
            # custom hand layer should be specified here
            raise NotImplementedError

        # Wrist pose
        self.cur_wrist_pos = hand_params.wrist_pos.clone()
        self.cur_wrist_rot = (hand_params.wrist_quat_wxyz if self.use_quat else hand_params.wrist_rot6d).clone()
        self.cur_wrist_pose = torch.eye(4).reshape(-1, 4, 4).repeat(self.nbatches, 1, 1).float().to(self.device)
        self.cur_wrist_pose[:, :3, 3] = self.cur_wrist_pos
        self.cur_wrist_pose[:, :3, :3] = roma.unitquat_to_rotmat(
            roma.quat_wxyz_to_xyzw(self.cur_wrist_rot)) if self.use_quat else (
            robust_compute_rotation_matrix_from_ortho6d(self.cur_wrist_rot))

        self.best_wrist_pos = self.cur_wrist_pos.clone()
        self.best_wrist_rot = self.cur_wrist_rot.clone()

        # Joint angles
        self.cur_joint_angles = hand_params.joint_angles
        self.best_joint_angles = self.cur_joint_angles.clone()

        # Initialize the optimizer, binding params
        self.opt_wrist_pos = torch.nn.Parameter(self.cur_wrist_pos.clone())
        self.opt_wrist_rot = torch.nn.Parameter(self.cur_wrist_rot.clone())
        joint_normalized = (hand_params.joint_angles - self.joint_means) / self.joint_ranges
        self.opt_joint_angles = torch.nn.Parameter(torch.atanh(joint_normalized.clamp(min=-1 + 1e-6, max=1 - 1e-6)))
        self.optimizer = torch.optim.AdamW([
            {'params': self.opt_wrist_pos, 'lr': 0.002},
            {'params': self.opt_wrist_rot, 'lr': 0.006},
            {'params': self.opt_joint_angles, 'lr': 0.001},
        ], lr=0.01)
        self.optimizing_scheduler = TorchStepLR(self.optimizer, step_size=50, gamma=0.9)

        # Joint limits
        self.joint_lower_limits = self.joint_means - self.joint_ranges
        self.joint_upper_limits = self.joint_means + self.joint_ranges

        self.index_thumb = torch.zeros(self.nbatches, dtype=torch.float32, device=self.device)
        self.middle_thumb = torch.zeros(self.nbatches, dtype=torch.float32, device=self.device)

        self.ncontacts = 6  # 4

        # Hand finger mask
        if "leap" in self.hand_model_name or "allegro" in self.hand_model_name:
            valid_mask = torch.tensor([
                True, True, True, True, True,  # Thumb
                True, True,  # [Palm]
                True, True, True, True, True,  # Index
                True,  # [Palm]
                True, True, True, True, True,  # Middle
                True, True,  # [Palm]
                True, True, True, True, True,  # Ring
                False, False,  # [Palm]
                False, False, False, False, False,  # Little
                True, False, True, False,  # Index  Side
                True, False, True, False,  # Middle Side
                True, False, True, False,  # Ring   Side
                False, False  # little
            ])
        elif ("shadow_hand" in self.hand_model_name or "svh_hand" in self.hand_model_name
              or "mano_hand" in self.hand_model_name):
            valid_mask = torch.tensor([
                True, True, True, True, True,  # Thumb
                True, True,  # [Palm]
                True, True, True, True, True,  # Index
                True,  # [Palm]
                True, True, True, True, True,  # Middle
                True, True,  # [Palm]
                True, True, True, True, True,  # Ring
                True, True,  # [Palm]
                True, True, True, True, True,  # Little
                True, False, True, False,  # Index  Side
                True, False, True, False,  # Middle Side
                True, False, True, False,  # Ring   Side
                True, False  # little
            ])
        else:
            raise NotImplementedError

        self.contact_idx = torch.tensor([
            0, 1, 2, 3, 4,  # Thumb
            5, 6,  # [Palm]
            7, 8, 9, 10, 11,  # Index
            12,  # [Palm]
            13, 14, 15, 16, 17,  # Middle
            18, 19,  # [Palm]
            20, 21, 22, 23, 24,  # Ring
            25, 26,  # [Palm]
            27, 28, 29, 30, 31,  # Little
            32, 33, 34, 35,  # Index  Side
            36, 37, 38, 39,  # Middle Side
            40, 41, 42, 43,  # Ring   Side
            44, 45,  # little
        ], dtype=torch.long).to(self.device)[valid_mask]

        # Make weights torch parameters
        if True:
            self.contact_weight = torch.tensor([
                0.5, 1, 1, 0.5, 0.5,  # Thumb
                1.0, 1.0,  # [Palm]
                0.5, 0.5, 0.5, 0.5, 0.5,  # Index
                1.0,  # [Palm]
                0.5, 0.5, 0.5, 0.5, 0.5,  # Middle
                1.0, 1.0,  # [Palm]
                0.5, 0.5, 0.5, 0.5, 0.5,  # Ring
                1.0, 1.0,  # [Palm]
                0.5, 0.5, 0.5, 0.5, 0.5,  # Little
                0.5, 0, 0.5, 0,  # Index  Side
                0.5, 0, 0.5, 0,  # Middle Side
                0.5, 0, 0.5, 0,  # Ring   Side
                0.5, 0,  # little
            ]).to(self.device)[valid_mask]
        else:
            # self.contact_weight = torch.ones(len(self.contact_idx)).to(self.device)
            pass

        self.force_closure_transf_matrix = torch.tensor([
            [0, 0, 0, 0, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, -1, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0]
        ], dtype=torch.float32, device=self.device)

        # Weights
        self.distance_weight = 100.0
        self.force_closure_weight = 50.0
        self.contact_align_weight = 0.5

    def _initialize_grasp_space(self, hand_layer: LeapHandLayer, out_hand_params: HandParams):
        """
        Initialize grasp translation, rotation, joint angles, and contact point indices

        Parameters
        ----------
        :param hand_layer: LeapHandLayer
        :param out_hand_params: Output hand params
        """

        # Initialize wrist grasp pose
        obj_dense_pcl, obj_pt_normals, obj_farthest_pts = self._initialize_object()
        if out_hand_params.wrist_pos is None or out_hand_params.wrist_quat_wxyz is None:
            grasp_rot, grasp_pos = self._calculate_grasp_pose_from_obj(obj_dense_pcl, obj_pt_normals, obj_farthest_pts)
            out_hand_params.wrist_pos = grasp_pos
            out_hand_params.wrist_rot6d = compute_rotation_ortho6d_from_matrix(grasp_rot)
            out_hand_params.wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(grasp_rot))

        # initialize joint angles
        if out_hand_params.joint_angles is None:
            # joint_angles_mu: hand-crafted canonicalized hand articulation
            # use truncated normal distribution to jitter the joint angles
            joint_angles_mu = hand_layer.get_init_angle()
            joint_angles_sigma = self.opt_params.jitter_strength * (hand_layer.joints_upper - hand_layer.joints_lower)
            joint_angles = torch.zeros([self.nbatches, hand_layer.n_dofs], dtype=torch.float32,
                                       device=self.device)
            for i in range(hand_layer.n_dofs):
                torch.nn.init.trunc_normal_(joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i],
                                            hand_layer.joints_lower[i] + 1e-6, hand_layer.joints_upper[i] - 1e-6)
            # joint_angles[:, [1, 5, 9]] = 0
            out_hand_params.joint_angles = joint_angles

    def _initialize_object(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_obj_dense_pcl = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        all_obj_pt_normals = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        all_obj_farthest_pts = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        for obj_mesh in self.object_data.meshes:
            # Create a layer (with inflating offset as ~0.01) for each object mesh
            # Get inflated convex hull
            obj_mesh.update_faces(obj_mesh.nondegenerate_faces())
            obj_mesh.remove_unreferenced_vertices()
            use_cvx = False
            if use_cvx:
                obj_mesh_ori = obj_mesh.convex_hull
                obj_vertices = obj_mesh_ori.vertices.copy()
                obj_faces = obj_mesh_ori.faces
                obj_vertices += 0.2 * obj_vertices / np.linalg.norm(obj_vertices, axis=1, keepdims=True)
                obj_postoffset_trimesh = trimesh.Trimesh(vertices=obj_vertices, faces=obj_faces).convex_hull
            else:
                # Offset the mesh
                obj_mesh_offset = 0.01
                obj_mesh_ori = obj_mesh.copy()
                obj_closed_mesh_ori = mrn.meshFromFacesVerts(obj_mesh_ori.faces, obj_mesh_ori.vertices)

                obj_mesh_offset_params = mrmeshpy.OffsetParameters()
                obj_mesh_offset_params.voxelSize = 0.01
                obj_postoffset_mesh = mrmeshpy.offsetMesh(obj_closed_mesh_ori, obj_mesh_offset, obj_mesh_offset_params)
                obj_postoffset_trimesh = trimesh.Trimesh(vertices=mrn.getNumpyVerts(obj_postoffset_mesh),
                                                         faces=mrn.getNumpyFaces(obj_postoffset_mesh.topology))

            obj_vertices = torch.tensor(obj_postoffset_trimesh.vertices, dtype=torch.float32, device=self.device)
            obj_faces = torch.tensor(obj_postoffset_trimesh.faces, dtype=torch.float32, device=self.device)
            obj_mesh_pytorch3d = pytorch3d.structures.Meshes(obj_vertices.unsqueeze(0), obj_faces.unsqueeze(0))

            # Sample obj mesh points (10 each batch)
            obj_pts_num = min(10000, 10 * self.nbatches)
            # [obj_mesh_pytorch3d] -> [obj_dense_pcl]
            obj_dense_pcl = pytorch3d.ops.sample_points_from_meshes(obj_mesh_pytorch3d, num_samples=obj_pts_num)

            # [obj_dense_pcl] -> [obj_farthest_pts]
            obj_farthest_pts = pytorch3d.ops.sample_farthest_points(obj_dense_pcl, K=self.nbatches)[0]

            # Squeeze [obj_farthest_pts, obj_dense_pcl], rem redundant dim
            obj_farthest_pts = obj_farthest_pts.squeeze(0)
            obj_dense_pcl = obj_dense_pcl.squeeze(0)

            # [obj_farthest_pts] -> [obj_closest_surface_pts]
            obj_closest_surface_pts, _, _ = obj_mesh_ori.nearest.on_surface(obj_farthest_pts.detach().cpu().numpy())
            obj_closest_surface_pts = torch.tensor(obj_closest_surface_pts, dtype=torch.float32, device=self.device)

            obj_pt_normals = (obj_closest_surface_pts - obj_farthest_pts) / (
                    obj_closest_surface_pts - obj_farthest_pts).norm(dim=1).unsqueeze(1)

            # Visualize [obj_mesh]'s pcl, normals, farthest-pts
            vis_obj_postoffset_trimesh = False
            if vis_obj_postoffset_trimesh:
                obj_farthest_pcl = trimesh.PointCloud(obj_farthest_pts.detach().cpu().numpy(), colors=(0, 255, 255))
                # create some rays
                ray_origins = obj_farthest_pts.detach().cpu().numpy()
                ray_directions = (obj_closest_surface_pts - obj_farthest_pts).detach().cpu().numpy()
                # stack rays into line segments for visualization as Path3D
                ray_visualize = trimesh.load_path(
                    np.hstack((ray_origins, ray_origins + ray_directions)).reshape(-1, 2, 3)
                )
                trimesh.Scene([obj_mesh_ori, obj_farthest_pcl, ray_visualize]).show()
                trimesh.Scene([obj_postoffset_trimesh]).show()

            # Group them all
            all_obj_dense_pcl = torch.cat([all_obj_dense_pcl, obj_dense_pcl])
            all_obj_pt_normals = torch.cat([all_obj_pt_normals, obj_pt_normals])
            all_obj_farthest_pts = torch.cat([all_obj_farthest_pts, obj_farthest_pts])

        return all_obj_dense_pcl, all_obj_pt_normals, all_obj_farthest_pts

    def _calculate_grasp_pose_from_obj(self, obj_dense_pcl, obj_pt_normals, obj_farthest_pts):
        # sample parameters
        distance = (self.opt_params.distance_lower + (self.opt_params.distance_upper - self.opt_params.distance_lower) *
                    torch.rand([self.nbatches],
                               dtype=torch.float32,
                               device=self.device))
        deviate_theta = (
                self.opt_params.joint_limit_lower + (
                self.opt_params.joint_limit_upper - self.opt_params.joint_limit_lower) *
                torch.rand([self.nbatches],
                           dtype=torch.float32,
                           device=self.device))
        process_theta = (
                self.opt_params.joint_limit_lower + (
                self.opt_params.joint_limit_upper - self.opt_params.joint_limit_lower) *
                torch.rand([self.nbatches],
                           dtype=torch.float32,
                           device=self.device))
        # solve transformation
        # grasp_rot_offset: rotate the hand to align its grasping direction with the +z axis
        # grasp_rot_local: jitter the hand's orientation in a cone
        # grasp_rot_global and grasp_pos: transform the hand to a position corresponding to point p sampled from the inflated convex hull

        grasp_rot_local = torch.zeros([self.nbatches, 3, 3], dtype=torch.float32, device=self.device)
        grasp_rot_global = torch.zeros([self.nbatches, 3, 3], dtype=torch.float32, device=self.device)
        radius = 0.05
        random_sign = np.random.choice([-1, 1])
        obj_dense_pcl_pts = obj_dense_pcl.cpu().numpy()
        for j in range(self.nbatches):
            grasp_rot_local[j] = torch.tensor(
                transforms3d.euler.euler2mat(process_theta[j], deviate_theta[j], 0, axes='sxyz'),
                dtype=torch.float32, device=self.device)

            distances = np.linalg.norm(obj_dense_pcl_pts - obj_farthest_pts[j].cpu().numpy(), axis=1)
            neighborhood_indices = np.where(distances < radius)[0]
            neighborhood_points = obj_dense_pcl_pts[neighborhood_indices]

            cov_matrix = np.cov(neighborhood_points.T)
            if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
                cov_matrix = np.nan_to_num(cov_matrix)

            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            sorted_indices = np.argsort(eigenvalues)[::-1]

            sorted_eigenvectors = eigenvectors[:, sorted_indices]

            grasp_rot_global[j] = torch.from_numpy(sorted_eigenvectors).to(self.device)

        # Transform based on z direction
        grasp_obj_mask = (grasp_rot_global[:, :3, 2] * obj_pt_normals).sum(dim=-1) < 0
        grasp_rot_global[grasp_obj_mask, :3, 0] *= -1
        grasp_rot_global[grasp_obj_mask, :3, 2] *= -1

        grasp_rot_offset_Z = 60 if random_sign == 1 else -120
        grasp_rot_offset = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 2, np.deg2rad(grasp_rot_offset_Z),
                                                                     axes='szxz'),
                                        dtype=torch.float32, device=self.device)

        grasp_rot = grasp_rot_global @ grasp_rot_local @ grasp_rot_offset
        grasp_pos = (
            # Move along +Z by [distance]
                obj_farthest_pts - distance.unsqueeze(1) * (
                grasp_rot_global @ grasp_rot_local @
                torch.tensor([0, 0, 1], dtype=torch.float32,
                             device=self.device).reshape(1, -1, 1)).squeeze(2) -
                # Then along +X
                (grasp_rot_global @ grasp_rot_offset @
                 torch.tensor([0.02, 0.00, 0],
                              dtype=torch.float32,
                              device=self.device).reshape(1, -1, 1)).squeeze(2))
        return grasp_rot, grasp_pos

    def sample_fingers_verts_and_normals(self, verts: torch.Tensor, vert_normals: torch.Tensor,
                                         down_sample_rate: int = 2):
        finger_verts = []
        finger_verts_normal = []
        split_indices = []
        count = 0

        split_indices.append(count)
        for key, value in self.finger_indices.items():
            finger_verts.append(verts[:, value][:, ::down_sample_rate])
            finger_verts_normal.append(vert_normals[:, value][:, ::down_sample_rate])
            count += finger_verts[-1].shape[1]
            split_indices.append(count)

        finger_verts = torch.cat(finger_verts, dim=1)
        finger_verts_normal = torch.cat(finger_verts_normal, dim=1)

        return finger_verts, finger_verts_normal, split_indices

    def compute_grasp_matrix(self, contact_points, contact_normals):
        batch_size, n_contacts, _ = contact_points.shape
        G = torch.zeros((batch_size, 6, n_contacts), dtype=contact_points.dtype, device=contact_points.device)

        G[:, :3, :] = contact_normals.permute(0, 2, 1)
        G[:, 3:, :] = torch.cross(contact_points, contact_normals).permute(0, 2, 1)

        return G

    def decode_joint_angles(self, with_limit: bool = False) -> torch.Tensor:
        if with_limit:
            output = (self.joint_means + self.joint_ranges * self.opt_joint_angles) % (2 * np.pi)
            output = torch.where(output > np.pi, output - 2 * np.pi, output)
            output = torch.clamp(output, min=self.joint_lower_limits, max=self.joint_upper_limits)
            return output
        else:
            assert not torch.isinf(
                torch.sum(self.opt_joint_angles)), f'{self.opt_joint_angles} contains an infinity value'
            return self.joint_means + self.joint_ranges * torch.tanh(self.opt_joint_angles)

    def compute_self_collision(self, verts: torch.Tensor, vert_normals: torch.Tensor):
        finger_verts, finger_verts_normal, splits = self.sample_fingers_verts_and_normals(verts, vert_normals,
                                                                                          down_sample_rate=2)
        self_collision_loss = None

        for i in range(1, len(splits) - 2):
            j2i_signed, i2j_signed, _, _, _, _ = point2point_signed(finger_verts[:, splits[i]:splits[i + 1]],
                                                                    finger_verts[:, splits[i + 1]:],
                                                                    finger_verts_normal[:, splits[i]:splits[i + 1]],
                                                                    finger_verts_normal[:, splits[i + 1]:])
            j2i_signed_dist_neg = torch.logical_and(j2i_signed.abs() < 0.01, j2i_signed < 0.0)
            i2j_signed_dist_neg = torch.logical_and(i2j_signed.abs() < 0.01, i2j_signed < 0.0)

            if self_collision_loss is None:
                self_collision_loss = torch.sum(i2j_signed * i2j_signed_dist_neg, dim=1)
            else:
                self_collision_loss += torch.sum(i2j_signed * i2j_signed_dist_neg, dim=1)
            self_collision_loss += torch.sum(j2i_signed * j2i_signed_dist_neg, dim=1)

        return self_collision_loss

    def compute_parallel_contact_loss(self, hand_anchors):
        left_points, right_points = self.parallel_contact_points[:, :3], self.parallel_contact_points[:, 3:]
        dis_left = torch.linalg.norm(left_points - hand_anchors[:, 9:12].mean(dim=1), dim=1)
        dis_right = torch.linalg.norm(right_points - hand_anchors[:, 2:5].mean(dim=1), dim=1)
        dis = dis_left + dis_right
        return dis * 1000

    def compute_close_distance(self, hand_anchors, h2o_signed):
        condition_0 = h2o_signed[:, self.hand_anchor_layer.vert_idx[2:5]].mean(dim=1) > 0.02
        thumb_index = torch.linalg.norm(hand_anchors[:, 2:5].mean(dim=1) - hand_anchors[:, 12], dim=1) * condition_0
        # condition_1 = h2o_signed[:, self.hand_anchor_layer.vert_idx[9:12]].mean(dim=1) > 0.025
        # index_thumb = torch.linalg.norm(hand_anchors[:, 9:12].mean(dim=1) - hand_anchors[:, 2:5].mean(dim=1), dim=1) * condition_1 * 0
        # condition_2 = h2o_signed[:, self.hand_anchor_layer.vert_idx[15:18]].mean(dim=1) > 0.025
        # middle_thumb = torch.linalg.norm(hand_anchors[:, 15:18].mean(dim=1) - hand_anchors[:, 1], dim=1) * condition_2 * 0

        # loss_close = (thumb_index + index_thumb + middle_thumb) * 500
        loss_close = thumb_index * 500
        return loss_close

    def update_opt_wrist_pose(self, wrist_pos: torch.Tensor, wrist_rot: torch.Tensor):
        with torch.no_grad():
            self.opt_wrist_pos.copy_(wrist_pos)
            self.opt_wrist_rot.copy_(wrist_rot if self.use_quat else
                                     roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(wrist_rot)))

    def forward(self, obstacle: ObjectData):
        """
        Implement loss function
        """
        # 1- Get [next_wrist_pose] as the optimized one [self.self.opt_wrist_pos/opt_wrist_rot]
        next_wrist_pose = torch.zeros_like(self.cur_wrist_pose, device=self.device)
        next_wrist_pos = self.opt_wrist_pos
        next_wrist_pose[:, :3, 3] = next_wrist_pos

        next_wrist_quat_wxyz = torch.nn.functional.normalize(self.opt_wrist_rot) if self.use_quat else None
        if self.use_quat:
            next_wrist_pose[:, :3, :3] = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(next_wrist_quat_wxyz))
        else:
            next_wrist_pose[:, :3, :3] = robust_compute_rotation_matrix_from_ortho6d(self.opt_wrist_rot)
        joint_angles = self.decode_joint_angles()

        # 2- Get Hand vertices/normals at the predicted [next_wrist_pose]
        if USE_MUJOCO_HAND_LAYER:
            if USE_MJ_WARP:
                assert self.use_quat
                self.hand_layer.ori_hand_p3d_meshes = mjw_geoms_to_pytorch3d_meshes(
                    self.hand_layer.mj_model,
                    self.hand_layer.mjw_model,
                    self.hand_layer.mjw_data,
                    body_names=self.hand_layer.hand_body_names,
                    hand_base_pose=torch.cat([next_wrist_pos.squeeze(), next_wrist_quat_wxyz.squeeze()]),
                    hand_qpos=joint_angles,
                    is_collision=self.hand_layer.use_collision_mesh,
                    body_p3d_meshes=self.hand_layer.ori_hand_p3d_meshes,
                    device=self.device
                )

                hand_p3dmesh_all = pytorch3d.structures.join_meshes_as_batch(
                    [mesh_meta[0] for _, mesh_meta in self.hand_layer.ori_hand_p3d_meshes.items()])
                pred_vertices = torch.cat(hand_p3dmesh_all.verts_list()).unsqueeze(0)
                pred_normals = torch.cat(hand_p3dmesh_all.verts_normals_list()).unsqueeze(0)
                # trimesh.Scene(p3d_to_trimesh(hand_p3dmesh_all)).show()
            else:
                pred_vertices, pred_normals = self.hand_layer.get_forward_vertices(next_wrist_pose, joint_angles)
        else:
            if USE_NEWTON_HAND_LAYER:
                assert self.use_quat
                next_wrist_pose = torch.cat([next_wrist_pos, roma.quat_wxyz_to_xyzw(next_wrist_quat_wxyz)], dim=-1)
            pred_vertices, pred_normals = self.hand_layer.get_forward_vertices(next_wrist_pose, joint_angles)

        self.hand_visual_verts = o3d_vox_downsample(pred_vertices.squeeze().detach().cpu().numpy())
        self.hand_visual_vert_normals = o3d_vox_downsample(pred_normals.squeeze().detach().cpu().numpy())
        next_grasp_site_pos = torch.matmul(next_wrist_pose,
                                           self.grasp_site_local_pos.transpose(0, 1)).transpose(1, 2)[..., :3]
        self.hand_visual_grasp_site_pos = next_grasp_site_pos.squeeze(dim=0).detach().cpu().numpy()

        # 2.1- Hand anchors of [pred_vertices/pred_normals] of hand at [next_wrist_pose]
        self.hand_anchors = self.hand_anchor_layer(pred_vertices)
        self.hand_anchors_normals = self.hand_anchor_layer(pred_normals)

        # 3- Losses at [next_wrist_pose]
        # 3.1- Collision with Env Obstacles (Floor, etc.)
        loss_collision_obstacle = 0
        if obstacle is not None:
            _, h2o_signed, _, _, _, _ = point2point_signed(
                pred_vertices, obstacle.all_points.repeat(self.nbatches, 1, 1), pred_normals,
                obstacle.all_normals.repeat(self.nbatches, 1, 1))

            h2o_dist_neg = torch.logical_and(h2o_signed.abs() < 0.05, h2o_signed < 0.0)
            loss_collision_obstacle = torch.sum(h2o_signed * h2o_dist_neg, dim=1) * -20
        # torch.cuda.synchronize()
        # time_start = time.time()

        # 3.2- Collision with Object
        object_points = self.object_data.all_points
        object_normals = self.object_data.all_normals
        o2h_signed, h2o_signed, obj_near_idx, hand_near_idx, o2h_vec, h2o_vec = point2point_signed(
            pred_vertices, object_points.repeat(self.nbatches, 1, 1),
            pred_normals, object_normals.repeat(self.nbatches, 1, 1),
        )

        o2h_dist_neg = torch.logical_and(o2h_signed.abs() < 0.005, o2h_signed < 0.0)
        h2o_dist_neg = torch.logical_and(h2o_signed.abs() < 0.005, h2o_signed < 0.0)

        loss_collision_h2o = torch.sum(h2o_signed * h2o_dist_neg, dim=1)
        loss_collision_o2h = torch.sum(o2h_signed * o2h_dist_neg, dim=1)

        hand_obj_collision = -20 * (1 * loss_collision_h2o + 10 * loss_collision_o2h)  # 75
        # hand_obj_collision = -200 * loss_collision_o2h  # 75
        # torch.cuda.synchronize()
        # time_cost = time.time() - time_start
        # print('time cost', time_cost)

        # 3.3- Self-Collision
        if self.hand_model_name == 'parallel_gripper':
            # No self collision with parallel jaw gripper
            hand_self_collision = 0
        else:
            hand_self_collision = -60 * self.compute_self_collision(pred_vertices, pred_normals)  # 60 as default

        # if iteration > 75:
        #     loss_close = self.compute_close_distance(hand_anchors, h2o_signed)
        # else:
        #     loss_close = 0

        # 3.4- Anchors' contact-with-object loss
        hand_near_idx = hand_near_idx[:, self.hand_anchor_layer.vert_idx][:, self.contact_idx]
        contact_hand_vec = torch.nn.functional.normalize(
            object_points[hand_near_idx] - self.hand_anchors[:, self.contact_idx], dim=-1)
        contact_obj_vec = -object_normals[hand_near_idx]

        contact_anchors_normals = self.hand_anchors_normals[:, self.contact_idx].view(-1, 1, 3)
        out_1 = torch.bmm(contact_anchors_normals, contact_hand_vec.view(-1, 3, 1)).view(self.nbatches, -1)
        out_2 = torch.bmm(contact_anchors_normals, contact_obj_vec.view(-1, 3, 1)).view(self.nbatches, -1)

        contact_align_loss = (1 - out_1).sum(-1) * 0.5 * self.contact_align_weight
        contact_align_loss += (1 - out_2).sum(-1) * 0.5 * self.contact_align_weight

        # 3.4.1- Anchors' contact distance loss
        # mask = (h2o_vec[:, self.hand_anchor_layer.vert_idx][:, self.contact_idx] * self.hand_anchors_normal[:, self.contact_idx]).sum(dim=-1) < 0
        E_dis = torch.sum(
            torch.abs(h2o_signed[:, self.hand_anchor_layer.vert_idx][:, self.contact_idx]) *
            self.contact_weight, dim=1) * self.distance_weight  # * self.config.contact_prob[i]

        # 3.4.2- E_fc: force closure loss
        if self.apply_force_closure:
            weights = torch.ones(len(self.contact_idx)).expand(self.nbatches, -1)
            select_contact_idx = torch.multinomial(weights, num_samples=self.ncontacts,
                                                   replacement=False).to(self.device)
            # select_contact_idx = torch.tensor([[2, 3, 4, 9, 10, 11]], dtype=torch.long).repeat(self.bs, 1).to(self.device)

            random_contact_idx = self.contact_idx[select_contact_idx]

            j = random_contact_idx.reshape(self.nbatches, self.ncontacts, 1)
            selected_anchors = self.hand_anchors[
                torch.arange(self.nbatches).reshape(self.nbatches, 1, 1), j, torch.arange(3)]

            obj_contact_normal = object_normals.squeeze()[hand_near_idx.gather(1, select_contact_idx)]

            contact_normal = obj_contact_normal.reshape(self.nbatches, 1, 3 * self.ncontacts)
            g = torch.cat([
                torch.eye(3, dtype=torch.float32, device=self.device).expand(self.nbatches, self.ncontacts, 3, 3)
                .reshape(self.nbatches, 3 * self.ncontacts, 3),
                (selected_anchors @ self.force_closure_transf_matrix).view(self.nbatches, 3 * self.ncontacts, 3)
            ], dim=2).float().to(self.device)
            norm = torch.norm(contact_normal @ g, dim=[1, 2])
            E_fc = norm * norm * self.force_closure_weight
        else:
            E_fc = 0

        # 3.5- Hand rotation-from-current loss
        if self.use_quat:
            hand_rot_loss = (1 - (next_wrist_quat_wxyz * self.cur_wrist_rot).sum(-1) ** 2)
        else:
            hand_rot_loss = roma.rotmat_geodesic_distance(next_wrist_pose[:, :3, :3],
                                                          self.cur_wrist_pose[:, :3, :3]) * 0.2

        # 3.6- Abnormal joint loss (distance from joint means -> hand-specific loss)
        angle_loss = self.hand_layer.compute_abnormal_joint_loss(joint_angles)

        if self.parallel_contact_points is not None:
            parallel_contact_loss = self.compute_parallel_contact_loss(self.hand_anchors)
        else:
            parallel_contact_loss = 0.0

        total_cost = (hand_obj_collision + hand_self_collision + E_dis + E_fc + contact_align_loss + hand_rot_loss
                      + loss_collision_obstacle + angle_loss + parallel_contact_loss)

        return total_cost

    def inference(self, return_anchors=False):
        with torch.no_grad():
            wrist_pose = torch.from_numpy(np.identity(4)).to(self.device).reshape(-1, 4, 4).float()
            wrist_pose[0, :3, :3] = roma.unitquat_to_rotmat(
                torch.nn.functional.normalize(roma.quat_wxyz_to_xyzw(self.best_wrist_rot))) if self.use_quat \
                else robust_compute_rotation_matrix_from_ortho6d(self.best_wrist_rot)
            wrist_pose[0, :3, 3] = self.best_wrist_pos
            pred_vertices, _ = self.hand_layer.get_forward_vertices(wrist_pose, self.best_joint_angles)
            pred_anchors = self.hand_anchor_layer(pred_vertices)
        if return_anchors:
            return pred_vertices, pred_anchors
        return pred_vertices

    def best_grasp_configuration(self, save_real=False) -> HandGrasp:
        # get best hand parameters
        assert self.best_wrist_rot is not None
        assert self.best_wrist_pos is not None
        assert self.best_joint_angles is not None

        if self.use_quat:
            wrist_roma_quat = torch.nn.functional.normalize(roma.quat_wxyz_to_xyzw(self.best_wrist_rot))
        else:
            wrist_roma_quat = roma.rotmat_to_unitquat(robust_compute_rotation_matrix_from_ortho6d(self.best_wrist_rot))
        wrist_pos = self.best_wrist_pos.clone()
        if save_real and self.use_mano_frame:
            mano_wrist_pose = torch.zeros_like(self.cur_wrist_pose, device=self.device)
            mano_wrist_pose[:, :3, :3] = roma.unitquat_to_rotmat(wrist_roma_quat)
            mano_wrist_pose[:, :3, 3] = self.best_wrist_pos.clone()
            mano_hand_pose = torch.matmul(mano_wrist_pose, self.hand_layer.base_2_world)
            wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(mano_hand_pose[:, :3, :3]))
            wrist_pos = mano_hand_pose[:, :3, 3]
        else:
            wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_roma_quat)

        return HandGrasp(wrist_quat=wrist_quat_wxyz.squeeze().cpu().numpy(),
                         wrist_pos=wrist_pos.squeeze().cpu().numpy(),
                         joint_angles=self.best_joint_angles.squeeze().cpu().numpy(),
                         obj_scale=self.object_data.scale,
                         obj_mesh_paths=self.object_data.mesh_paths)

    def last_grasp_configuration(self, save_real=False) -> HandGrasp:
        # get current hand parameters
        if self.use_quat:
            wrist_roma_quat = torch.nn.functional.normalize(roma.quat_wxyz_to_xyzw(self.best_wrist_rot))
        else:
            wrist_roma_quat = roma.rotmat_to_unitquat(robust_compute_rotation_matrix_from_ortho6d(self.best_wrist_rot))
        wrist_pos = self.opt_wrist_pos.detach()

        if save_real and self.use_mano_frame:
            mano_wrist_pose = torch.zeros_like(self.cur_wrist_pose, device=self.device)
            mano_wrist_pose[:, :3, :3] = roma.unitquat_to_rotmat(wrist_roma_quat)
            mano_wrist_pose[:, :3, 3] = wrist_pos.clone()
            mano_hand_pose = torch.matmul(mano_wrist_pose, self.hand_layer.base_2_world)
            wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(mano_hand_pose[:, :3, :3]))
            wrist_pos = mano_hand_pose[:, :3, 3]
        else:
            wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_roma_quat)

        return HandGrasp(wrist_quat=wrist_quat_wxyz.squeeze().cpu().numpy(),
                         wrist_pos=wrist_pos.squeeze().cpu().numpy(),
                         joint_angles=self.decode_joint_angles(with_limit=False).detach().squeeze().cpu().numpy(),
                         obj_scale=self.object_data.scale,
                         obj_mesh_paths=self.object_data.mesh_paths)

    def optimize(self, cur_wrist_pos: Optional[torch.Tensor] = None, cur_wrist_rot: Optional[torch.Tensor] = None,
                 obstacle: ObjectData = None, n_iters=1000):
        min_loss = 1e8

        # Update [self.cur_wrist_pose]
        if cur_wrist_pos is not None:
            self.cur_wrist_pos.copy_(cur_wrist_pos)
            self.cur_wrist_pose[:, :3, 3] = cur_wrist_pos
        if cur_wrist_rot is not None:
            self.cur_wrist_rot.copy_(cur_wrist_rot)
            self.cur_wrist_pose[:, :3, :3] = roma.unitquat_to_rotmat(
                roma.quat_wxyz_to_xyzw(cur_wrist_rot)) if self.use_quat else (
                robust_compute_rotation_matrix_from_ortho6d(cur_wrist_rot))

        # Start optimizing
        iter_range = tqdm(range(n_iters + 1), desc='hand optimizing process') if self.nbatches > 1 \
            else range(n_iters + 1)
        for iter_step in iter_range:
            loss = self.forward(obstacle)

            if iter_step >= 0:
                loss_mask = loss < min_loss
                min_loss = torch.where(loss_mask, loss, min_loss)
                nonzero_idx = torch.nonzero(loss_mask, as_tuple=True)[0]
                if not torch.numel(nonzero_idx) == 0:
                    with (torch.no_grad()):
                        opt_wrist_rot = self.opt_wrist_rot[nonzero_idx]
                        self.best_wrist_rot[nonzero_idx] = (torch.nn.functional.normalize(opt_wrist_rot, dim=1)
                                                            if self.use_quat else opt_wrist_rot).clone().detach()
                        self.best_wrist_pos[nonzero_idx] = self.opt_wrist_pos[nonzero_idx].clone().detach()
                        self.best_joint_angles[nonzero_idx] = \
                            self.decode_joint_angles(with_limit=False)[nonzero_idx].clone().detach()
            self.optimizer.zero_grad()
            loss.mean().backward()
            # if iter_step % 20 == 0:
            #     self.theta.grad *= 0
            # else:
            #     self.wrist_pos.grad *= 0
            #     self.wrist_rot.grad *= 0

            self.optimizer.step()
            self.optimizing_scheduler.step()

            # print(self.optimizer.state_dict())
            # print('{}-th iter: {}'.format(iter_step, loss.mean().item()))

    def step_optimize(self, cur_wrist_pos: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      cur_wrist_rot: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      cur_obj_mesh_poses: Optional[list[Union[np.ndarray, torch.Tensor]]] = None,
                      cur_obst_mesh_poses: Optional[list[Union[np.ndarray, torch.Tensor]]] = None,
                      substeps_num: int = 1) -> HandGrasp:
        # Transform wrist
        with torch.no_grad():
            if cur_wrist_pos is not None:
                cur_wrist_pos = torch.tensor(cur_wrist_pos, device=self.device)

            if cur_wrist_rot is not None:
                if self.use_quat:
                    cur_wrist_rot = torch.tensor(cur_wrist_rot, device=self.device)
                else:
                    cur_wrist_rot = torch.tensor(cur_wrist_rot.transpose(1, 2)[:, :2].reshape(-1, 6),
                                                 device=self.device)

        # Transform objects & obstacles
        if cur_obj_mesh_poses:
            self.object_data.transform_to(cur_obj_mesh_poses)

        if cur_obst_mesh_poses and self.obstacle_data:
            self.obstacle_data.transform_to(cur_obst_mesh_poses)

        # Next optimal grasp
        self.optimize(cur_wrist_pos, cur_wrist_rot,
                      obstacle=self.obstacle_data, n_iters=substeps_num)

        return self.best_grasp_configuration(save_real=False)

    def visualize_grasp(self, grasp: HandGrasp, object_meshes: list[trimesh.Trimesh]):
        # Init grasp
        cur_theta = self.cur_joint_angles.reshape(-1, self.hand_layer.n_dofs)

        # CURRENT GRASP
        cur_wrist_pose = torch.zeros_like(self.cur_wrist_pose, device=self.device)
        if self.use_quat:
            cur_wrist_pose[:, :3, :3] = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(self.cur_wrist_rot))
        else:  # use rot6d representation
            cur_wrist_pose[:, :3, :3] = robust_compute_rotation_matrix_from_ortho6d(self.cur_wrist_rot)
        cur_wrist_pose[:, :3, 3] = self.cur_wrist_pos
        cur_verts, cur_verts_normals = self.hand_layer.get_forward_vertices(cur_wrist_pose, cur_theta)

        # NEXT OPTIMAL GRASP & HAND ANCHORS
        next_wrist_pose = torch.zeros_like(self.cur_wrist_pose, device=self.device)
        next_theta = torch.from_numpy(grasp.joint_angles).to(self.device).reshape(-1, self.hand_layer.n_dofs)
        next_wrist_pose[:, :3, :3] = roma.unitquat_to_rotmat(
            roma.quat_wxyz_to_xyzw(torch.from_numpy(grasp.wrist_quat)).to(self.device))
        next_wrist_pose[:, :3, 3] = torch.from_numpy(grasp.wrist_pos).to(self.device)
        next_verts, next_verts_normals = self.hand_layer.get_forward_vertices(next_wrist_pose, next_theta)
        next_anchors = self.hand_anchor_layer.forward(next_verts)

        for idx in range(self.nbatches):
            next_pcl = trimesh.PointCloud(next_verts[idx].squeeze().cpu().numpy(), colors=(0, 255, 255))
            next_pcl_anchor = trimesh.PointCloud(next_anchors[idx].squeeze().cpu().numpy(), colors=(255, 0, 0))
            cur_pcl = trimesh.PointCloud(cur_verts[idx].squeeze().cpu().numpy(), colors=(255, 0, 255))
            scene = trimesh.Scene([cur_pcl, next_pcl, next_pcl_anchor] + object_meshes)
            scene.show()
