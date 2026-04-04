# leap_hand layer for torch
from typing import Optional, Union
import torch
from torch.autograd import gradcheck as torch_gradcheck
import numpy as np
import warp as wp

import coacd
import trimesh

import mujoco as mj
import mujoco_warp as mjw
import pytorch_kinematics as pk
import pytorch3d.structures
from pytorch3d.transforms import matrix_to_quaternion as p3d_matrix_to_quaternion
import roma

# judo
from judo import PACKAGE_ROOT
from judo.hand_layers.leap_layer import LeapHandLayer, LeapAnchor

# mjmanip
from mjmanip.utils import IDENTITY_POSE, mj_step, mj_draw_pointcloud
from mjmanip.trimesh_utils import mj_get_body_trimeshes
from mjmanip.warp_utils import wp_transform_from_mj, wp_kernel_transform_mesh_points, wp_kernel_compute_vertex_normals
from mjmanip.pytorch3d_utils import mjw_geoms_to_pytorch3d_meshes
from mjmanip.control.fabrics.prod.kinematics import Kinematics as TorchWarpKinematics
from mjmanip.control.fabrics.taskmaps.robot_frame_origins_taskmap import RobotKinematics as RobotTorchWarpKinematics

USE_MJ_WARP = False  # NOTE: GPU MJW Mesh fetch is not correct yet!
LEAP_LAYER_HAND_ASSETS_DIR = f"{PACKAGE_ROOT}/hand_layers/leap_hand_layer/assets"

# judo
from judo.tasks.panda_leap_pick import USE_LEAP_MJX

if USE_LEAP_MJX:
    from mjmanip.robot.leap_mjx import LEAP_ASSETS_DIR, HAND_MODEL_DIR as LEAP_HAND_MODEL_DIR, HAND_XML_PATH, LeapMjx

    LEAP = LeapMjx
else:
    from mjmanip.robot.leap import LEAP_ASSETS_DIR, HAND_MODEL_DIR as LEAP_HAND_MODEL_DIR, HAND_XML_PATH, Leap

    LEAP = Leap


# All lengths are in mm and rotations in radians


class MJLeapHandLayer(LeapHandLayer):
    def __init__(self, hand_model_desc: str,
                 hand_base_pose: np.ndarray,
                 joint_angles: np.ndarray,
                 batch_size: int = 1,
                 to_mano_frame: bool = False, show_mesh: bool = False,
                 use_collision_mesh: bool = False,
                 regen_cache: bool = False,
                 visualized: bool = False,
                 device: str = 'cuda'):
        super().__init__(hand_model_desc, hand_base_pose, joint_angles, batch_size, to_mano_frame, show_mesh,
                         use_collision_mesh=use_collision_mesh, regen_cache=regen_cache,
                         visualized=visualized, device=device)
        assert hand_model_desc.endswith('.xml')

    def init_kinematics(self):
        self.mj_spec, self.chain = pk.build_chain_from_mjcf(self.hand_model_desc, device=self.device)
        self.mj_spec.meshdir = LEAP_ASSETS_DIR
        if False:
            # Actually, no need, hand_base will be teleported anyway later in [get_forward_vertices()]
            hand_base_spec = self.mj_spec.body(LEAP.HAND_BASE_NAME)
            hand_base_spec.pos = self.hand_base_pose.squeeze()[:3]
            hand_base_spec.quat = self.hand_base_pose.squeeze()[3:]
        self.mj_model: mj.MjModel = self.mj_spec.compile()
        self.mj_data: mj.MjData = mj.MjData(self.mj_model)
        self.mj_hand_base = self.mj_model.body(LEAP.HAND_BASE_NAME)
        self.hand_body_names = [self.mj_model.body(i).name for i in range(self.mj_model.nbody)]
        self.hand_body_names.remove('world')

        # MJW
        self.mjw_model = mjw.put_model(self.mj_model)
        self.mjw_data = mjw.put_data(self.mj_model, self.mj_data, nworld=1, njmax=300)

        self.torch_warp_kinematics = TorchWarpKinematics(self.hand_model_desc, self.batch_size,
                                                         thread_across_links=False,
                                                         device=self.device,
                                                         robot_base_transform=wp_transform_from_mj(
                                                             self.hand_base_pose.squeeze()),
                                                         frame_names=self.hand_body_names)

        self.torch_warp_link_idxs = torch.tensor(
            [self.torch_warp_kinematics.get_link_index(body_name) for body_name in self.hand_body_names],
            device=self.device)

        # Joints
        self.joint_lowers = self.chain.low
        self.joint_uppers = self.chain.high
        self.joint_means = (self.joint_lowers + self.joint_uppers) / 2
        self.joint_ranges = self.joint_means - self.joint_lowers
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_dofs = self.chain.n_joints  # only used here for robot hand with no mimic joint

        # Original geom meshes: No convex hull + Not globally posed!
        self.ori_hand_meshes: dict[str, tuple[trimesh.Trimesh, np.ndarray]] = (
            mj_get_body_trimeshes(self.mj_model,
                                  model_spec=self.mj_spec,
                                  body_names=self.hand_body_names,
                                  is_collision=self.use_collision_mesh,
                                  use_global_pose=False))

        if USE_MJ_WARP:
            self.ori_hand_p3d_meshes: dict[str, tuple[pytorch3d.structures.Meshes, pytorch3d.transforms.Transform3d]] = \
                mjw_geoms_to_pytorch3d_meshes(self.mj_model,
                                              self.mjw_model, self.mjw_data,
                                              body_names=self.hand_body_names,
                                              is_collision=self.use_collision_mesh,
                                              device=self.device)

        # Warp geom mesh data
        self.ori_geom_meshes_points: dict[str, torch.Tensor] = {}
        self.ori_geom_meshes_point_normals: dict[str, torch.Tensor] = {}

        # Create cache data
        regen_cache = self.regen_cache or not (self.geom_convex_meshes and
                                               self.hand_surface_points and self.visible_point_indices)
        if regen_cache:
            # Create [self.hand_surface_points, self.visible_point_indices]
            self.create_assets()
        else:
            # [self.hand_surface_points, self.visible_point_indices] must have been loaded from cache folders earlier
            assert len(self.hand_surface_points)
            assert len(self.visible_point_indices)
            self.make_contact_points = False

        # Create mesh verts/normals
        self.ori_geom_meshes_points, self.ori_geom_meshes_point_normals = (
            self.create_mesh_verts_normals(self.hand_surface_points, self.visible_point_indices))

        # Fetch hand segment indices
        self.hand_segment_indices, self.hand_finger_indices = self.get_hand_segment_indices()

    def create_assets(self):
        '''
        To create needed assets for the first running.
        Should run before first use.
        '''
        # 1- Convex meshes
        for geom_name, geom_mesh_meta in self.ori_hand_meshes.items():
            geom_mesh = geom_mesh_meta[0]
            use_convex_hull = True or 'palm' in geom_name
            if use_convex_hull:
                geom_mesh = geom_mesh.convex_hull
            else:
                # 'coacd'
                coacd_threshold = 0.3
                mesh = coacd.Mesh(geom_mesh.vertices, geom_mesh.faces)
                parts = coacd.run_coacd(mesh, threshold=coacd_threshold, apx_mode='ch')
                meshes = []
                for part in parts:
                    mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
                    meshes.append(mesh)
                geom_mesh = trimesh.boolean.boolean_manifold(meshes, 'union')
            self.geom_convex_meshes[geom_name] = geom_mesh
        self.save_geom_convex_meshes(self.geom_convex_meshes)

        # 3.1- Sample [self.hand_composite_points] + [self.hand_surface_points] from GLOBALLY POSED MESHES
        self.hand_composite_points, self.hand_surface_points, hand_whole_mesh = (
            self.sample_composite_points(self.hand_base_pose, self.joint_angles))

        # 3.2- Sample visible composite points
        self.visible_point_indices = self.sample_visible_points(hand_whole_mesh, self.hand_composite_points,
                                                                down_sampling=self.use_collision_mesh)
        if self.visualized:
            trimesh.Scene(hand_whole_mesh).show()
            # Visualize hand surface points, which are still GLOBAL now
            self.visualize_hand_surface(self.hand_surface_points, self.visible_point_indices)

        # 4. LOCALIZE [self.hand_surface_points]
        # NOTE: Since [self.hand_surface_points] was originally saved in global world frame
        # => Here, need to transform [wp_meshes_points/normals] back to the local geom frames
        wp_meshes_points, wp_meshes_normals = self.create_mesh_verts_normals(self.hand_surface_points, to_wp=True)
        for geom_name, geom_surface_points in self.hand_surface_points.items():
            mj_body = self.mj_data.body(self.mj_model.geom(geom_name).bodyid[0])
            geom_local_pose = self.ori_hand_meshes[geom_name][1]

            wp.launch(kernel=wp_kernel_transform_mesh_points,
                      dim=len(wp_meshes_points[geom_name]),
                      inputs=[wp_meshes_points[geom_name],
                              wp.transform_inverse(wp_transform_from_mj(np.concat([mj_body.xpos, mj_body.xquat])) *
                                                   wp_transform_from_mj(geom_local_pose)),
                              wp.vec3(1, 1, 1)],
                      outputs=[wp_meshes_points[geom_name]],
                      device=str(self.device))

            wp.launch(kernel=wp_kernel_transform_mesh_points,
                      dim=len(wp_meshes_normals[geom_name]),
                      inputs=[wp_meshes_normals[geom_name],
                              wp.transform_inverse(wp_transform_from_mj(np.concat([mj_body.xpos, mj_body.xquat])) *
                                                   wp_transform_from_mj(geom_local_pose)),
                              wp.vec3(1, 1, 1)],
                      outputs=[wp_meshes_normals[geom_name]],
                      device=str(self.device))

            self.hand_surface_points[geom_name] = np.concatenate([wp_meshes_points[geom_name].numpy(),
                                                                  wp_meshes_normals[geom_name].numpy()], axis=-1)
        # Save [self.hand_surface_points]
        self.save_surface_points(self.hand_surface_points)
        print("Leap hand layer asset caches created!")

    @classmethod
    def create_mesh_verts_normals(cls, hand_surface_points: dict[str, np.ndarray],
                                  visible_point_indices: Optional[dict[str, np.ndarray]] = None,
                                  to_wp: bool = False,
                                  device: Optional[str] = 'cuda') \
            -> tuple[dict[str, Union[wp.array, torch.Tensor]], dict[str, Union[wp.array, torch.Tensor]]]:
        mesh_points = {}
        mesh_normals = {}
        # [self.hand_surface_points[visible_point_indices]] -> [self.ori_geom_meshes_points/ori_geom_meshes_point_normals]
        for geom_name, geom_surface_points in hand_surface_points.items():
            geom_visible_indices = visible_point_indices.get(geom_name, None) if visible_point_indices else None
            geom_surface = geom_surface_points[geom_visible_indices] if (geom_visible_indices is not None
                                                                         and geom_visible_indices.any()) \
                else geom_surface_points

            if to_wp:
                mesh_points[geom_name] = wp.from_numpy(geom_surface[:, :3], dtype=wp.vec3, device=device)
                mesh_normals[geom_name] = wp.from_numpy(geom_surface[:, 3:6], dtype=wp.vec3, device=device)
            else:
                temp = torch.ones(geom_surface.shape[0], 1).float().to(device=device)
                mesh_points[geom_name] = torch.cat([torch.from_numpy(geom_surface[:, :3]).to(device=device), temp],
                                                   dim=-1).float()
                mesh_normals[geom_name] = torch.cat([torch.from_numpy(geom_surface[:, 3:6]).to(device=device), temp],
                                                    dim=-1).float()
        return mesh_points, mesh_normals

    def sample_composite_points(self, hand_base_pose: np.ndarray, hand_qpos: np.ndarray, n_points: int = int(2e+4)) \
            -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], trimesh.Trimesh]:
        global_posed_meshes = mj_get_body_trimeshes(self.mj_model,
                                                    data=self.mj_data,
                                                    model_spec=self.mj_spec,
                                                    body_names=self.hand_body_names,
                                                    base_pose=hand_base_pose.squeeze(),
                                                    qpos=hand_qpos.squeeze(),
                                                    is_collision=self.use_collision_mesh,
                                                    use_convex_hull=True,
                                                    use_global_pose=True)

        hand_composite_points = {}
        hand_surface_points = {}
        hand_meshes = []
        for geom_name, geom_mesh_meta in global_posed_meshes.items():
            geom_mesh = geom_mesh_meta[0]
            hand_meshes.append(geom_mesh)

            # Sample [self.hand_surface_points]
            points, point_normals = self.sample_geom_surface_points(geom_mesh, n_points=n_points)
            hand_composite_points[geom_name] = points
            hand_surface_points[geom_name] = np.concatenate([points, point_normals], axis=-1)

        return hand_composite_points, hand_surface_points, trimesh.util.concatenate(hand_meshes)

    def step_forward(self, hand_base_pose: np.ndarray, hand_qpos: np.ndarray):
        # NOTE: NOT Differentiable, for reference only & benchmark with MJWarp or TorchWarpKinematics!
        hand_base_pose = hand_base_pose.squeeze()
        self.mj_hand_base.pos[:] = hand_base_pose[:3]
        self.mj_hand_base.quat[:] = hand_base_pose[3:]
        self.mj_data.qpos[:] = hand_qpos.squeeze()
        mj.mj_fwdKinematics(self.mj_model, self.mj_data)
        mj.mj_fwdPosition(self.mj_model, self.mj_data)

    def step_forward_diff(self, hand_base_pose: torch.Tensor, hand_qpos: torch.Tensor) -> torch.Tensor:
        # Calculate the link transforms and their origin Jacobians.
        # TODO: Make [RobotTorchWarpKinematics.apply()] differentiable!
        self.torch_warp_kinematics.robot_base_transform = wp_transform_from_mj(hand_base_pose.squeeze())
        link_transforms, jacobians = RobotTorchWarpKinematics.apply(hand_qpos, self.torch_warp_kinematics)
        return link_transforms

    def get_forward_vertices(self, hand_base_pose: torch.Tensor, hand_qpos: torch.Tensor):
        """
        NOTE: This function must be differentiable, so purely written in Torch!
        """
        frame_tfs = self.forward(hand_qpos)

        verts = []
        verts_normal = []

        # for mesh_key, mesh in self.meshes.items():
        for geom_name, geom_mesh_meta in self.ori_hand_meshes.items():
            geom_body_name = self.mj_model.body(self.mj_model.geom(geom_name).bodyid[0]).name
            geom_link = self.chain.find_link(geom_body_name)
            # geom_trimesh = geom_mesh_meta[0]
            # geom_local_pose = geom_mesh_meta[1]

            # geom_global_pose = hand_base_pose * (frame_pose * geom_offset)
            frame_pose = frame_tfs[geom_body_name].get_matrix()
            torch.matmul(self.to_mano_transform, frame_pose) if self.to_mano_frame else frame_pose
            pk_geom = next((g for g in (geom_link.collisions if self.use_collision_mesh else geom_link.visuals)
                            if g.name == geom_name))
            assert pk_geom, f"pytorch_kinematics chain: {geom_name} is not found in pk's collisions/visuals!"
            geom_global_pose = torch.matmul(hand_base_pose, torch.matmul(frame_pose, pk_geom.offset.get_matrix()))

            # hand_base_pose * link_pose * link_geom_verts/normals
            vertices = self.ori_geom_meshes_points[geom_name]
            batch_vertices = torch.matmul(geom_global_pose, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
            verts.append(batch_vertices)
            vertex_normals = self.ori_geom_meshes_point_normals[geom_name]
            geom_global_pose[:, :3, 3] *= 0
            batch_vertex_normals = (
                torch.matmul(geom_global_pose, vertex_normals.transpose(0, 1)).transpose(1, 2)[..., :3])
            verts_normal.append(batch_vertex_normals)

        verts = torch.cat(verts, dim=1).contiguous()
        verts_normal = torch.cat(verts_normal, dim=1).contiguous()
        return verts, verts_normal

    """
    NOTE: KEPT FOR REF ONLY!
    def get_forward_vertices(self, hand_base_pose: torch.Tensor, hand_qpos: torch.Tensor):
        #TODO: Make this function differentiable in torch!
        body_pos_xyzws = {}
        link_transforms = self.step_forward_diff(hand_base_pose, hand_qpos)
        link_positions = link_transforms[:, self.torch_warp_link_idxs, :3]
        link_quat_xyzws = link_transforms[:, self.torch_warp_link_idxs, 3:]
        for body_name in self.hand_body_names:
            body_link_idx = self.torch_warp_kinematics.get_link_index(body_name)
            body_pos_xyzws[body_name] = torch.cat([link_positions[:, body_link_idx],
                                                   link_quat_xyzws[:, body_link_idx]], dim=-1).squeeze()

        verts = []
        verts_normals = []
        for geom_name, geom_mesh_meta in self.ori_hand_meshes.items():
            geom_body_name = self.mj_model.body(self.mj_model.geom(geom_name).bodyid[0]).name
            wp_body_pose = wp.transform(*body_pos_xyzws[geom_body_name])
            # geom_trimesh = geom_mesh_meta[0]
            geom_local_pose = geom_mesh_meta[1]

            # Vertices
            # NOTE: Transform from the original [geom_trimesh]'s [mesh_points]
            mesh_points = wp.zeros(len(self.ori_geom_meshes_points[geom_name]), dtype=wp.vec3, device=self.device)
            wp.launch(kernel=wp_kernel_transform_mesh_points,
                      dim=len(mesh_points),
                      inputs=[self.ori_geom_meshes_points[geom_name],
                              wp_body_pose *
                              wp_transform_from_mj(geom_local_pose),
                              wp.vec3(1, 1, 1)],
                      outputs=[mesh_points],
                      device=str(self.device))

            mesh_verts = wp.to_torch(mesh_points)
            verts.append(mesh_verts)

            # Normals
            normals = wp.zeros(len(mesh_points), dtype=wp.vec3, device=self.device)
            wp.launch(kernel=wp_kernel_transform_mesh_points,
                      dim=len(mesh_points),
                      inputs=[self.ori_geom_meshes_point_normals[geom_name],
                              wp_body_pose *
                              wp_transform_from_mj(geom_local_pose),
                              wp.vec3(1, 1, 1)],
                      outputs=[normals],
                      device=str(self.device))
            verts_normals.append(wp.to_torch(normals))

        verts = torch.concatenate(verts).unsqueeze(0).float().contiguous()
        verts_normals = torch.concatenate(verts_normals).unsqueeze(0).float().contiguous()
        return verts, verts_normals
    """

    @classmethod
    def visualize_hand_surface(cls, hand_surface_points: dict[str, np.ndarray],
                               visible_point_indices: dict[str, np.ndarray]) -> None:
        surface_points_normals = np.concatenate(
            [sp[visible_point_indices[g]] for g, sp in hand_surface_points.items() if
             g in visible_point_indices], axis=0
        )
        surface_points = surface_points_normals[:, :3]
        trimesh.Scene(trimesh.PointCloud(surface_points, colors=(0, 255, 255))).show()
        surface_normals = surface_points_normals[:, 3:6]
        # surface_normals /= np.maximum(np.linalg.norm(surface_normals, axis=1, keepdims=True), 1e-8)
        surface_normal_arrows = (np.stack([surface_points, surface_points + surface_normals * 0.005], axis=1)
                                 .reshape(-1, 3))
        normal_segs = trimesh.load_path(surface_normal_arrows)
        trimesh.Scene(normal_segs).show()


if __name__ == "__main__":
    from loop_rate_limiters import RateLimiter
    import math
    from mujoco import viewer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pick_anchor_points = True
    hand_qpos = np.zeros((1, LEAP.HAND_DOFS_NO), dtype=np.float32)
    hand_base_pose = np.array([0, 0, 5, 1, 0, 0, 0])[np.newaxis]
    mj_hand_layer = MJLeapHandLayer(hand_model_desc=HAND_XML_PATH,
                                    hand_base_pose=hand_base_pose,
                                    joint_angles=hand_qpos,
                                    use_collision_mesh=False,
                                    regen_cache=True,
                                    visualized=True,
                                    device=device)
    mj_hand_layer.make_contact_points = False
    model = mj_hand_layer.mj_model
    data = mj_hand_layer.mj_data
    rate = RateLimiter(frequency=1 / model.opt.timestep, warn=False)
    with mj.viewer.launch_passive(model=model, data=data, show_left_ui=False,
                                  show_right_ui=False) as mj_viewer:
        mj.mjv_defaultFreeCamera(model, mj_viewer.cam)
        i = 0.0
        while mj_viewer.is_running():
            mj.mj_camlight(model, data)

            i += 1.1
            if False:
                print(torch_gradcheck(mj_hand_layer.step_forward_diff, (
                    torch.tensor([math.sin(i) * 0.5, math.cos(i) * 0.5, 3, 1, 0, 0, 0],
                                 device=device).unsqueeze(
                        0).double().requires_grad_(True),
                    torch.rand(1, LEAP.HAND_DOFS_NO, device=device).double().requires_grad_(True))))
            verts, normals = mj_hand_layer.get_forward_vertices(
                hand_base_pose=torch.tensor([math.sin(i) * 0.5, math.cos(i) * 0.5, 3, 1, 0, 0, 0],
                                            device=device).unsqueeze(0).requires_grad_(True),
                hand_qpos=torch.rand(1, LEAP.HAND_DOFS_NO, device=device).requires_grad_(True))
            mj_draw_pointcloud(mj_viewer.user_scn, verts.detach().cpu().numpy().squeeze())

            if pick_anchor_points:
                anchor_layer = LeapAnchor()
                anchor_layer.pick_points(verts.squeeze().cpu().numpy())

            # Step
            mj_step(model, data, kinematics_only=True)
            mj_viewer.sync()
            rate.sleep()
        mj_viewer.close()
