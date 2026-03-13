# leap_hand layer for torch
import torch
import trimesh
import os
import numpy as np
import coacd

from mesh_to_sdf import get_surface_point_cloud
from scipy.spatial import KDTree
import point_cloud_utils as pcu

import mujoco as mj
import mujoco_warp as mjw
import pytorch3d.structures
import pytorch_kinematics as pk

# judo
from judo import PACKAGE_ROOT

# mjmanip
from mjmanip.utils import IDENTITY_POSE, mj_get_geom_mesh_meta
from mjmanip.trimesh_utils import mj_geoms_to_trimeshes
from mjmanip.pytorch3d_utils import mjw_geoms_to_pytorch3d_meshes
from mjmanip.robot.leap_mjx import HAND_MODEL_DIR as LEAP_HAND_MODEL_DIR, LeapMjx

LEAP_LAYER_HAND_ASSETS_DIR = f"{PACKAGE_ROOT}/hand_layers/leap_hand_layer/assets"


# All lengths are in mm and rotations in radians

def mj_geom_mesh_path(mj_model: mj.MjModel, geom_name: str):
    geom_id = mj_model.geom(geom_name).id
    mesh_path, mesh_scale = mj_get_geom_mesh_meta(mj_model, geom_id)
    return mesh_path


class MJLeapHandLayer(torch.nn.Module):
    def __init__(self, hand_model_desc: str,
                 to_mano_frame=True, show_mesh=False,
                 device='cuda'):
        super().__init__()

        self.show_mesh = show_mesh
        self.use_collision_mesh = False
        self.to_mano_frame = to_mano_frame
        self.device = device
        self.name = 'leap_hand'
        self.finger_num = 4

        self.BASE_DIR = os.path.split(os.path.abspath(__file__))[0]
        self.hand_model_desc = hand_model_desc
        self.is_from_mjcf = hand_model_desc.endswith('.xml')
        self.mj_spec, self.chain = pk.build_chain_from_mjcf(hand_model_desc, device)
        self.mj_model = self.mj_spec.compile()
        self.mj_data = mj.MjData(self.mj_model)
        self.mjw_model = mjw.put_model(self.mj_model)
        self.mjw_data = mjw.put_data(self.mj_model, self.mj_data, nworld=1, njmax=300)
        self.hand_body_names = [self.mj_model.body(i).name for i in range(self.mj_model.nbody)]
        self.hand_body_names.remove('world')

        self.joints_lower = self.chain.low
        self.joints_upper = self.chain.high
        self.joints_mean = (self.joints_lower + self.joints_upper) / 2
        self.joints_range = self.joints_mean - self.joints_lower
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_dofs = self.chain.n_joints  # only used here for robot hand with no mimic joint

        # transformation for align the robot hand to mano hand frame, used for
        self.to_mano_transform = torch.eye(4).to(torch.float32).to(device)
        if self.to_mano_frame:
            self.to_mano_transform[:3, :] = torch.tensor([[-1, 0, 0, 0],
                                                          [0, 0, 1, 0.0175],
                                                          [0, 1, 0, 0.0375]])

        self.register_buffer('base_2_world', self.to_mano_transform)

        self.ordered_finger_endeffort = ['leap_mount_collision_0', 'leap_mount_collision_1',
                                         'th_ds_collision', 'if_ds_collision',
                                         'mf_ds_collision', 'rf_ds_collision'] if self.use_collision_mesh else [
            'leap_mount_visual', 'th_ds_visual',
            'if_ds_visual', 'mf_ds_visual', 'rf_ds_visual']

        self.ori_hand_meshes: dict[str, dict[str, tuple[trimesh.Trimesh, np.ndarray]]] = {}
        self.ori_hand_p3d_meshes: dict[str, tuple[pytorch3d.structures.Meshes, pytorch3d.transforms.Transform3d]] = {}
        self.hand_cvx_meshes: dict[str, trimesh.Trimesh] = {}
        self.meshes_meta: dict[str, tuple[trimesh.Trimesh, torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.make_contact_points = True
        self.create_assets()
        self.hand_segment_indices, self.hand_finger_indices = self.get_hand_segment_indices()

    def create_assets(self):
        '''
        To create needed assets for the first running.
        Should run before first use.
        '''
        theta = np.zeros((1, self.n_dofs), dtype=np.float32)

        # Hand meshes
        if False:
            self.ori_hand_p3d_meshes = mjw_geoms_to_pytorch3d_meshes(
                self.mj_model,
                self.mjw_model, self.mjw_data,
                body_names=self.hand_body_names,
                hand_base_pose=torch.from_numpy(IDENTITY_POSE),
                hand_qpos=torch.from_numpy(theta),
                is_collision=self.use_collision_mesh,
                device=self.device
            )
        self.ori_hand_meshes = mj_geoms_to_trimeshes(self.mj_model, self.mj_data,
                                                     body_names=self.hand_body_names,
                                                     hand_base_pose=IDENTITY_POSE,
                                                     hand_qpos=theta,
                                                     is_collision=self.use_collision_mesh)
        hand_meshes = {_: m[0] for _, m in self.ori_hand_meshes.items()}
        # trimesh.Scene([self.ori_hand_meshes]).show()
        self.hand_cvx_meshes = self.save_part_meshes(hand_meshes,
                                                     dst_dir=f"{LEAP_LAYER_HAND_ASSETS_DIR}/hand_meshes_cvx")
        self.hand_mesh_points = self.sample_points_on_mesh(self.hand_cvx_meshes,
                                                           dst_dir=f"{LEAP_LAYER_HAND_ASSETS_DIR}/hand_points")

        # SAMPLE hand_composite_points
        hand_cvx_mesh = trimesh.util.concatenate(self.hand_cvx_meshes.values())
        self.hand_mesh_visible_points = self.sample_visible_points(hand_meshes=[hand_cvx_mesh],
                                                                   dst_dir=f"{LEAP_LAYER_HAND_ASSETS_DIR}/visible_point_indices")
        print("Assets created!")

    def get_hand_segment_indices(self):
        hand_segment_indices = {}
        hand_finger_indices = {}
        segment_start = 0  # torch.tensor(0, dtype=torch.long, device=self.device)
        finger_start = 0  # torch.tensor(0, dtype=torch.long, device=self.device)
        for mesh_name, mesh in self.ori_hand_meshes.items():
            # torch.tensor(self.meshes[link_name][1].shape[0], dtype=torch.long, device=self.device)
            end = segment_start + mesh[0].vertices.shape[0]
            hand_segment_indices[mesh_name] = torch.arange(segment_start, end)  # [segment_start, end]
            if mesh_name in self.ordered_finger_endeffort:
                hand_finger_indices[mesh_name] = torch.arange(finger_start, end)  # [finger_start, end]
                finger_start = end  # end.clone()
            segment_start = end  # end.clone()
        return hand_segment_indices, hand_finger_indices

    def forward(self, theta):
        """
        Args:
            theta (Tensor (batch_size x 15)): The degrees of freedom of the Robot hand.
       """
        ret = self.chain.forward_kinematics(theta)
        if self.is_from_mjcf:
            ret.pop("world")
        return ret

    def compute_abnormal_joint_loss(self, theta):
        loss_1 = -torch.clamp(theta[:, 5] - theta[:, 1], -1, 0) * 10
        loss_2 = -torch.clamp(theta[:, 9] - theta[:, 5], -1, 0) * 10
        loss_3 = torch.abs(theta[:, [2, 3, 6, 7, 10, 11]] - self.joints_mean[[2, 3, 6, 7, 10, 11]].unsqueeze(0)).sum(
            dim=-1) * 2
        return loss_1 + loss_2 + loss_3

    def get_init_angle(self):
        init_angle = (self.joints_upper - self.joints_lower) / 6.0 + self.joints_lower
        init_angle[1] = -0.1
        init_angle[5] = 0.0
        init_angle[9] = 0.1
        init_angle[12] = 0.8
        return init_angle

    def get_hand_mesh(self, pose, ret):
        bs = pose.shape[0]

        meshes = []
        for key, _ in ret.items():
            rotmat = ret[key].get_matrix()
            rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, rotmat))

            if key in self.meshes_meta:
                vertices = self.meshes_meta[key][0]
                batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
                face = self.meshes_meta[key][1]
                sub_meshes = [trimesh.Trimesh(vertices.cpu().numpy(), face) for vertices in batch_vertices]
                meshes.append(sub_meshes)

        hand_meshes = []
        for j in range(bs):
            hand = [meshes[i][j] for i in range(len(meshes))]
            hand_mesh = np.sum(hand)
            hand_meshes.append(hand_mesh)
        return hand_meshes

    def get_forward_hand_mesh(self, pose, theta):
        outputs = self.forward(theta)

        hand_meshes = self.get_hand_mesh(pose, outputs)

        return hand_meshes

    def get_forward_vertices_mujoco(self, hand_base_pose, hand_qpos):
        cur_hand_meshes = mj_geoms_to_trimeshes(self.mj_model, self.mj_data,
                                                body_names=self.hand_body_names,
                                                hand_base_pose=hand_base_pose, hand_qpos=hand_qpos,
                                                is_collision=self.use_collision_mesh,
                                                body_trimeshes=self.ori_hand_meshes)
        hand_mesh_all = trimesh.util.concatenate([m[0] for _, m in cur_hand_meshes.items()])
        verts = torch.tensor(hand_mesh_all.vertices, device=self.device).unsqueeze(0).float()
        verts_normal = torch.tensor(hand_mesh_all.vertex_normals, device=self.device).unsqueeze(0).float()
        return verts, verts_normal

    def save_part_meshes(self, hand_meshes: dict[str, trimesh.Trimesh], dst_dir: str, method='convexhull',
                         threshold=0.3,
                         save_to_disk: bool = False) -> dict[str, trimesh.Trimesh]:
        cvx_meshes = {}
        for mesh_name, mesh in hand_meshes.items():
            if 'palm' in mesh_name:
                tmp_method = 'convexhull'
            else:
                tmp_method = method
            if tmp_method == 'coacd':
                mesh = coacd.Mesh(mesh.vertices, mesh.faces)
                parts = coacd.run_coacd(mesh, threshold=threshold, apx_mode='ch')
                meshes = []
                for part in parts:
                    mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
                    meshes.append(mesh)
                new_mesh = trimesh.boolean.boolean_manifold(meshes, 'union')
            elif tmp_method == 'convexhull':
                new_mesh = mesh.convex_hull
            else:
                raise ValueError('method must be coacd or convexhull')

            cvx_meshes[mesh_name] = new_mesh
            if save_to_disk:
                os.makedirs(dst_dir, exist_ok=True)
                new_mesh_path = os.path.join(dst_dir, f"{mesh_name}.stl")
                print('cvx mesh saved to path:', new_mesh_path)
                new_mesh.export(new_mesh_path)
        if save_to_disk:
            print('cvx meshes saving: finished')
        return cvx_meshes

    def sample_points_on_mesh(self, cvx_meshes: dict[str, trimesh.Trimesh], dst_dir: str,
                              save_to_disk: bool = False) -> dict[str, np.ndarray]:
        np.random.seed(0)
        all_mesh_points = {}
        for mesh_name, convex_mesh in cvx_meshes.items():
            points, idx = trimesh.sample.sample_surface_even(convex_mesh, 50000, radius=None)
            point_normals = convex_mesh.face_normals[idx]
            vis = False
            if vis:
                pc = trimesh.PointCloud(points, colors=(255, 255, 0))
                ray_visualization = trimesh.load_path(np.hstack((points,
                                                                 points + point_normals / 100)).reshape(-1, 2, 3))
                trimesh.Scene([pc, ray_visualization]).show()

            mesh_points = np.concatenate([points, point_normals], axis=-1)
            all_mesh_points[mesh_name] = mesh_points
            if save_to_disk:
                os.makedirs(dst_dir, exist_ok=True)
                mesh_points_path = os.path.join(dst_dir, f"{mesh_name}.npy")
                np.save(mesh_points_path, mesh_points)
                print('mesh points saved to path:', mesh_points_path)
        if save_to_disk:
            print('mesh points saving: finished')
        return all_mesh_points

    def sample_visible_points(self, hand_meshes: list[trimesh.Trimesh],
                              dst_dir: str, voxel_size=0.0055, save_to_disk: bool = False) -> dict[str, np.ndarray]:
        count = 0

        results = [get_surface_point_cloud(hand_mesh, scan_count=100, scan_resolution=400)
                   for hand_mesh in hand_meshes]
        hand_points = np.concatenate([result.points for result in results])
        if self.use_collision_mesh:
            hand_points = pcu.downsample_point_cloud_on_voxel_grid(voxel_size / 2, hand_points)
            hand_points = pcu.downsample_point_cloud_on_voxel_grid(voxel_size / 2, hand_points)

        visible_points: dict[str, np.ndarray] = {}
        for mesh_name, point_info in self.hand_mesh_points.items():
            hand_point_tree = KDTree(data=hand_points)
            dist, index = hand_point_tree.query(point_info[:, :3], k=1)
            mask = dist < 0.0005
            index = index[mask]

            masked_points = point_info[mask]
            v_sampled = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, masked_points[:, :3])
            v_sampled = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, v_sampled)

            # v_sampled = o3d_vox_downsample(masked_points, 0.005)
            count += len(v_sampled)
            # trimesh.PointCloud(v_sampled, colors=(255, 255, 0)).show()

            _, v_sampled_idx = KDTree(point_info[:, :3]).query(v_sampled, k=1)
            # trimesh.PointCloud(point_info[:, :3][v_sampled_idx]).show()
            hand_points[index] = np.array([-1e6, -1e6, 1e6])
            visible_points[mesh_name] = v_sampled

            if save_to_disk:
                hand_visible_points_path = f"{dst_dir}/{mesh_name}.npy"
                os.makedirs(dst_dir, exist_ok=True)
                np.save(hand_visible_points_path, v_sampled_idx)
                print("visible points saved:", hand_visible_points_path)
        if save_to_disk:
            print("visible points saving: finished")
        return visible_points
