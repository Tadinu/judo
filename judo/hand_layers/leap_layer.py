# leap_hand layer for torch
from typing import Union
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

import mujoco as mj
import roma

from .layer_asset_utils import LEAP_LAYER_CACHE_DIR

# mjmanip
from mjmanip.utils import mj_get_mesh_file_path
from mjmanip.trimesh_utils import mj_geom_spec_to_trimesh

# judo
from judo.tasks.panda_leap_pick import USE_LEAP_MJX

if USE_LEAP_MJX:
    from mjmanip.robot.leap_mjx import LEAP_ASSETS_DIR, LeapMjx

    LEAP = LeapMjx
else:
    from mjmanip.robot.leap import LEAP_ASSETS_DIR, Leap

    LEAP = Leap


# All lengths are in mm and rotations in radians


class LeapHandLayer(torch.nn.Module):
    def __init__(self, hand_model_desc: str,
                 hand_base_pose: np.ndarray,
                 joint_angles: np.ndarray,
                 to_mano_frame: bool = False, show_mesh: bool = False, hand_type: str = 'right',
                 use_collision_mesh=False, regen_cache: bool = False, device: str = 'cuda'):
        super().__init__()

        self.show_mesh = show_mesh
        self.use_collision_mesh = use_collision_mesh
        self.to_mano_frame = to_mano_frame
        self.device = device
        self.name = 'leap_hand'
        self.hand_type = hand_type
        self.finger_num = 4

        self.is_from_urdf = hand_model_desc.endswith('urdf')
        if self.is_from_urdf:
            self.chain = pk.build_chain_from_urdf(open(hand_model_desc).read()).to(device=device)
        else:
            self.mj_spec, self.chain = pk.build_chain_from_mjcf(hand_model_desc, device=device)
            self.mj_spec.meshdir = LEAP_ASSETS_DIR
            hand_base_spec = self.mj_spec.body(LEAP.HAND_BASE_NAME)
            hand_base_spec.pos = hand_base_pose[:3]
            hand_base_spec.quat = hand_base_pose[3:]
            self.mj_model: mj.MjModel = self.mj_spec.compile()
            self.mj_data: mj.MjData = mj.MjData(self.mj_model)
            self.hand_body_names = [self.mj_model.body(i).name for i in range(self.mj_model.nbody)]

        self.hand_base_pose = torch.eye(4).reshape(-1, 4, 4).float()
        self.hand_base_pose[:, :3, 3] = torch.from_numpy(hand_base_pose[:3]).to(device).float()
        self.hand_base_pose[:, :3, :3] = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(
            torch.from_numpy(hand_base_pose[3:]).to(device).float()))

        self.joint_angles = torch.from_numpy(joint_angles).to(device).float()
        self.joint_lowers = self.chain.low
        self.joint_uppers = self.chain.high
        self.joint_means = (self.joint_lowers + self.joint_uppers) / 2
        self.joint_ranges = self.joint_means - self.joint_lowers
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_dofs = self.chain.n_joints  # only used here for robot hand with no mimic joint

        self.regen_cache = regen_cache
        self.geom_meshes = {}
        self.geom_convex_meshes = self.load_assets(f"{LEAP_LAYER_CACHE_DIR}/hand_meshes_cvx", is_mesh=True)
        self.hand_points = self.load_assets(f"{LEAP_LAYER_CACHE_DIR}/hand_points")
        self.visible_point_indices = self.load_assets(f"{LEAP_LAYER_CACHE_DIR}/visible_point_indices")
        self.hand_composite_points = {}
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

        if self.regen_cache or not (os.path.exists(f'{LEAP_LAYER_CACHE_DIR}/hand_meshes_cvx')
                                    and os.path.exists(f'{LEAP_LAYER_CACHE_DIR}/hand_points')
                                    and os.path.exists(f'{LEAP_LAYER_CACHE_DIR}/visible_point_indices')
                                    and os.path.exists(f'{LEAP_LAYER_CACHE_DIR}/hand.obj')
                                    and os.path.exists(f'{LEAP_LAYER_CACHE_DIR}/hand_all_zero.obj')
        ):
            self.create_assets()
        else:
            if self.to_mano_frame:
                self.to_mano_transform[:3, :] = torch.tensor([[-1, 0, 0, 0],
                                                              [0, 0, 1, 0.0175],
                                                              [0, 1, 0, 0.0375]]).float().to(self.device)
            self.make_contact_points = False
            self.meshes = self.load_meshes()

        self.hand_segment_indices, self.hand_finger_indices = self.get_hand_segment_indices()
        self.register_buffer('base_2_world', self.to_mano_transform)

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
        theta = self.joint_angles.unsqueeze(0)

        show_mesh = self.show_mesh
        self.show_mesh = True  # mesh with face
        self.make_contact_points = True  # True: creating convex meshes

        self.meshes = self.load_meshes()
        self.save_geom_convex_meshes()
        self.sample_surface_points()

        hand_meshes = self.get_forward_hand_mesh(pose, theta)[0]
        hand_parts = hand_meshes.split()

        hand_single_mesh = trimesh.boolean.boolean_manifold(hand_parts, 'union') if self.is_from_urdf else \
            trimesh.util.concatenate(hand_parts)
        hand_single_mesh.export(f'{LEAP_LAYER_CACHE_DIR}/hand.obj')

        self.show_mesh = True
        self.make_contact_points = False
        self.meshes = self.load_meshes()
        hand_all_zero_mesh = self.get_forward_hand_mesh(pose, theta)[0]
        hand_all_zero_mesh.export(f'{LEAP_LAYER_CACHE_DIR}/hand_all_zero.obj')

        self.show_mesh = False
        self.make_contact_points = True
        self.meshes = self.load_meshes()

        self.get_forward_vertices(pose, theta)  # Sample [hand_composite_points]
        self.sample_visible_points(hand_single_mesh)  # Sample [visible_points] from [hand_composite_points]
        self.show_mesh = True
        self.make_contact_points = False
        self.meshes = self.load_meshes()
        hand_to_mano_mesh = self.get_forward_hand_mesh(pose, theta)[0]
        hand_to_mano_mesh.export(f'{LEAP_LAYER_CACHE_DIR}/hand_to_mano_frame.obj')

        self.make_contact_points = False
        self.show_mesh = show_mesh

    def load_meshes(self):
        meshes = {}
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
                self.geom_meshes[geom_name] = geom_mesh

                # 2- Turn [geom_mesh] to convex mesh, saving it to [self.geom_convex_meshes]
                if self.make_contact_points:
                    threshold = 0.3
                    use_convex_hull = True or 'palm' in geom_name
                    if use_convex_hull:
                        geom_mesh = geom_mesh.convex_hull
                    else:
                        # 'coacd'
                        mesh = coacd.Mesh(geom_mesh.vertices, geom_mesh.faces)
                        parts = coacd.run_coacd(mesh, threshold=threshold, apx_mode='ch')
                        meshes = []
                        for part in parts:
                            mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
                            meshes.append(mesh)
                        geom_mesh = trimesh.boolean.boolean_manifold(meshes, 'union')
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
                    meshes[geom_name] = [
                        torch.cat((geom_verts, temp), dim=-1).to(self.device),
                        geom_mesh.faces,
                        torch.cat((geom_vertex_normals, temp), dim=-1).float().to(self.device)
                    ]
                else:
                    if geom_name not in self.hand_points:
                        points, point_normals = self.sample_geom_surface_points(geom_mesh)
                        self.hand_points[geom_name] = np.concatenate([points, point_normals], axis=-1)

                    points_info = self.hand_points[geom_name]
                    if self.make_contact_points:
                        idxs = np.arange(len(points_info))
                    else:
                        idxs = self.visible_point_indices[geom_name]

                    geom_verts = torch.from_numpy(points_info[idxs, :3]).float().to(self.device)
                    geom_verts = geom_pre_transform.transform_points(geom_verts)

                    geom_vertex_normals = torch.from_numpy(points_info[idxs, 3:6]).float().to(self.device)
                    geom_vertex_normals = geom_pre_transform.transform_normals(geom_vertex_normals)
                    temp = torch.ones(idxs.shape[0], 1).float()
                    meshes[geom_name] = [
                        torch.cat((geom_verts, temp), dim=-1).to(self.device),
                        torch.zeros([0]),  # no real meaning, just for placeholder
                        torch.cat((geom_vertex_normals, temp), dim=-1).float().to(self.device)
                    ]
        return meshes

    def get_hand_segment_indices(self):
        hand_segment_indices = {}
        hand_finger_indices = {}
        segment_start = 0  # torch.tensor(0, dtype=torch.long, device=self.device)
        finger_start = 0  # torch.tensor(0, dtype=torch.long, device=self.device)
        for geom_name_list in self.link_geom_names.values():
            geom_name_list = [geom_name_list] if isinstance(geom_name_list, str) else geom_name_list
            for geom_name in geom_name_list:
                if geom_name not in self.meshes:
                    continue
                end = segment_start + self.meshes[geom_name][0].shape[
                    0]  # torch.tensor(self.meshes[link_name][0].shape[0], dtype=torch.long, device=self.device)
                hand_segment_indices[geom_name] = torch.arange(segment_start, end)  # [segment_start, end]
                if geom_name in self.ordered_finger_endeffort:
                    hand_finger_indices[geom_name] = torch.arange(finger_start, end)  # [finger_start, end]
                    finger_start = end  # end.clone()
                segment_start = end  # end.clone()
        return hand_segment_indices, hand_finger_indices

    def forward(self, theta):
        """
        Args:
            theta (Tensor (batch_size x 15)): The degrees of freedom of the Robot hand.
       """
        ret = self.chain.forward_kinematics(theta)
        return ret

    def compute_abnormal_joint_loss(self, theta):
        loss_1 = -torch.clamp(theta[:, 5] - theta[:, 1], -1, 0) * 10
        loss_2 = -torch.clamp(theta[:, 9] - theta[:, 5], -1, 0) * 10
        loss_3 = torch.abs(theta[:, [2, 3, 6, 7, 10, 11]] - self.joint_means[[2, 3, 6, 7, 10, 11]].unsqueeze(0)).sum(
            dim=-1) * 2
        return loss_1 + loss_2 + loss_3

    def get_init_angle(self):
        init_angle = (self.joint_uppers - self.joint_lowers) / 6.0 + self.joint_lowers
        init_angle[1] = -0.1
        init_angle[5] = 0.0
        init_angle[9] = 0.1
        init_angle[12] = 0.8
        return init_angle

    def get_hand_mesh(self, pose, frame_tfs: dict[str, tf.Transform3d]) -> trimesh.Trimesh:
        bs = pose.shape[0]

        meshes = []
        for link_key, geom_name_list in self.link_geom_names.items():
            geom_name_list = [geom_name_list] if isinstance(geom_name_list, str) else geom_name_list
            for geom_name in geom_name_list:
                mesh_data = self.meshes[geom_name]
                rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, frame_tfs[link_key].get_matrix()))
                vertices = mesh_data[0]
                batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
                face = mesh_data[1]
                sub_meshes = [trimesh.Trimesh(vertices.cpu().numpy(), face) for vertices in batch_vertices]
                meshes.append(sub_meshes)

        hand_meshes = []
        for j in range(bs):
            hand_mesh = np.sum([meshes[i][j] for i in range(len(meshes))])
            hand_meshes.append(hand_mesh)
        # trimesh.Scene(hand_meshes).show()
        return hand_meshes

    def get_forward_hand_mesh(self, pose: torch.Tensor, theta: torch.Tensor) -> trimesh.Trimesh:
        frame_tfs = self.forward(theta)
        return self.get_hand_mesh(pose, frame_tfs)

    def get_forward_vertices(self, pose: torch.Tensor, theta: torch.Tensor):
        frame_tfs = self.forward(theta)

        verts = []
        verts_normal = []

        # for mesh_key, mesh in self.meshes.items():
        for link_key, geom_names in self.link_geom_names.items():
            geom_name_list = [geom_names] if isinstance(geom_names, str) else geom_names
            for geom_name in geom_name_list:
                if geom_name not in self.meshes:
                    continue
                mesh_data = self.meshes[geom_name]
                rotmat = frame_tfs[link_key].get_matrix()
                rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, rotmat))

                vertices = mesh_data[0]
                vertex_normals = mesh_data[2]
                batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
                verts.append(batch_vertices)

                if self.make_contact_points:
                    self.hand_composite_points[geom_name] = batch_vertices.squeeze().detach().cpu().numpy()
                rotmat[:, :3, 3] *= 0
                batch_vertex_normals = torch.matmul(rotmat, vertex_normals.transpose(0, 1)).transpose(1, 2)[..., :3]
                verts_normal.append(batch_vertex_normals)

        verts = torch.cat(verts, dim=1).contiguous()
        verts_normal = torch.cat(verts_normal, dim=1).contiguous()
        return verts, verts_normal

    def save_geom_convex_meshes(self, dst=f'{LEAP_LAYER_CACHE_DIR}/hand_meshes_cvx'):
        for geom_name, convex_mesh in self.geom_convex_meshes.items():
            os.makedirs(dst, exist_ok=True)
            filepath = f"{dst}/{geom_name}.stl"
            print('save to path:', filepath)
            convex_mesh.export(filepath)
        print('saving convex meshes finished')

    def sample_visible_points(self, hand_mesh: trimesh.Trimesh, voxel_size=0.0055, down_sampling: bool = False):
        count = 0

        # hand = trimesh.load(f'{LEAP_LAYER_CACHE_DIR}/hand.obj', force='mesh')
        result = get_surface_point_cloud(hand_mesh, scan_count=100, scan_resolution=200)
        points = np.array(result.points)
        if down_sampling:
            points = pcu.downsample_point_cloud_on_voxel_grid(voxel_size / 2, points)
            points = pcu.downsample_point_cloud_on_voxel_grid(voxel_size / 2, points)

        for geom_name, point_info in self.hand_composite_points.items():
            point_tree = KDTree(data=points)
            dist, index = point_tree.query(point_info[:, :3], k=1)
            mask = dist < 0.0005
            index = index[mask]

            visible_points = point_info[mask]
            v_sampled = visible_points[:, :3]
            if down_sampling:
                v_sampled = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, v_sampled)
                v_sampled = pcu.downsample_point_cloud_on_voxel_grid(voxel_size, v_sampled)

            # v_sampled = o3d_vox_downsample(visible_points, 0.005)
            count += len(v_sampled)
            # pc = trimesh.PointCloud(v_sampled, colors=(255, 255, 0))
            # pc.show()

            _, v_sampled_idx = KDTree(point_info[:, :3]).query(v_sampled, k=1)
            # pc = trimesh.PointCloud(point_info[:, :3][v_sampled_idx])
            # pc.show()
            points[index] = np.array([-1e6, -1e6, 1e6])
            os.makedirs(f'{LEAP_LAYER_CACHE_DIR}/visible_point_indices', exist_ok=True)
            np.save(f'{LEAP_LAYER_CACHE_DIR}/visible_point_indices/{geom_name}.npy', v_sampled_idx)
            self.visible_point_indices[geom_name] = v_sampled_idx
        print(count)

    def sample_geom_surface_points(self, geom_mesh: trimesh.Trimesh, n_points: int = 10000) -> tuple[
        np.ndarray, np.ndarray]:
        np.random.seed(0)
        points, idx = trimesh.sample.sample_surface_even(geom_mesh, n_points, radius=None)
        point_normals = geom_mesh.face_normals[idx]
        return points, point_normals

    def sample_surface_points(self, dst_dir=f'{LEAP_LAYER_CACHE_DIR}/hand_points', n_points: int = 50000,
                              visualize: bool = False):
        for geom_name, geom_convex_mesh in self.geom_convex_meshes.items():
            if geom_name in self.hand_points:
                points_info = self.hand_points[geom_name]
            else:
                points, point_normals = self.sample_geom_surface_points(geom_convex_mesh, n_points)
                points_info = np.concatenate([points, point_normals], axis=-1)

            os.makedirs(dst_dir, exist_ok=True)
            np.save(os.path.join(dst_dir, f"{geom_name}.npy"), points_info)

            if visualize:
                pc = trimesh.PointCloud(points, colors=(255, 255, 0))
                ray_visualization = trimesh.load_path(np.hstack((points,
                                                                 points + point_normals / 100)).reshape(-1, 2, 3))
                trimesh.Scene([pc, ray_visualization]).show()


class LeapAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # vert_idx
        vert_idx = np.array([
            # thumb finger
            1382, 1522, 1541, 1667, 1493,
            428, 179,

            # index finger
            1806, 2289, 2408, 2405, 2442,  # 2324
            19,

            # middle finger
            2504, 3016, 3164, 3049, 3060,
            364, 626,

            # ring finger
            3454, 3756, 3863, 3844, 3915,
            0, 0,  # place holder

            # little finger
            0, 0, 0, 0, 0,  # place holder

            # # plus
            2420, 2332, 2131, 2241,  # 2440  2463
            3129, 3133, 2895, 3005,
            3815, 3778, 3644, 3713,
            0, 0,  # place holder

        ])
        # vert_idx = np.load(os.path.join(self.BASE_DIR, 'anchor_idx.npy'))
        self.register_buffer("vert_idx", torch.from_numpy(vert_idx).long())

    def forward(self, vertices):
        """
        vertices: TENSOR[N_BATCH, 4040, 3]
        """
        anchor_pos = vertices[:, self.vert_idx, :]
        return anchor_pos

    def pick_points(self, vertices: np.ndarray):
        import open3d as o3d
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print(vis.get_picked_points())
        return vis.get_picked_points()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    show_mesh = False
    to_mano_frame = True
    hand = LeapHandLayer(show_mesh=show_mesh, to_mano_frame=to_mano_frame, device=device)

    pose = torch.eye(4).to(device).reshape(-1, 4, 4).float().to(device)
    theta = np.zeros((1, 16), dtype=np.float32)
    theta[0, :4] = np.array([0.0, 0.0, 0, 0])
    theta = torch.from_numpy(theta).to(device)

    # mesh version
    if show_mesh:
        hand_mesh = hand.get_forward_hand_mesh(pose, theta)[0]
        hand_mesh.show()
    else:
        hand_verts, hand_normals = hand.get_forward_vertices(pose, theta)
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
