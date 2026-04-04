# leap_hand layer in Newton mesh
from pathlib import Path

import warp as wp
import torch
import numpy as np
import trimesh

import newton

# judo
from judo import PACKAGE_ROOT
from judo.hand_layers.leap_layer import LeapHandLayer

# mjmanip
from mjmanip.utils import IDENTITY_POSE, mj_get_geom_mesh_meta
from mjmanip.trimesh_utils import mj_get_body_trimeshes
from mjmanip.pytorch3d_utils import mjw_geoms_to_pytorch3d_meshes
from mjmanip.newton.newton_backend import NewtonBackend
from mjmanip.warp_utils import wp_transform_from_mj, wp_kernel_transform_mesh_points, wp_kernel_compute_vertex_normals
from mjmanip.newton.newton_utils import (NewtonShapeMeta, nt_get_links_shapes_meta,
                                         nt_get_body_transform)

LEAP_LAYER_HAND_ASSETS_DIR = f"{PACKAGE_ROOT}/hand_layers/leap_hand_layer/assets"

# judo
from judo.tasks.panda_leap_pick import USE_LEAP_MJX

if USE_LEAP_MJX:
    from mjmanip.robot.leap_mjx import HAND_MODEL_DIR as LEAP_HAND_MODEL_DIR, LeapMjx, HAND_XML_PATH

    LEAP = LeapMjx
else:
    from mjmanip.robot.leap import HAND_MODEL_DIR as LEAP_HAND_MODEL_DIR, Leap, HAND_XML_PATH

    LEAP = Leap


# All lengths are in mm and rotations in radians
class NTLeapHandLayer(LeapHandLayer):
    def __init__(self, hand_model_desc: str,
                 hand_base_pose: np.ndarray,
                 joint_angles: np.ndarray,
                 batch_size: int = 1,
                 to_mano_frame: bool = True, show_mesh: bool = False,
                 use_collision_mesh: bool = False,
                 regen_cache: bool = False,
                 visualized: bool = False,
                 device: str = 'cuda'):
        super().__init__(hand_model_desc, hand_base_pose, joint_angles, batch_size, to_mano_frame, show_mesh,
                         use_collision_mesh=use_collision_mesh, regen_cache=regen_cache,
                         visualized=visualized, device=device)

    def init_kinematics(self):
        self.nt_backend: NewtonBackend = self.create_newton_backend(self.hand_model_desc, self.hand_base_pose)
        self.nt_model_builder = self.nt_backend.model_builder
        self.hand_body_names = self.nt_model_builder.body_label
        self.joint_lowers = torch.tensor(self.nt_model_builder.joint_limit_lower, device=self.device).float()
        self.joint_uppers = torch.tensor(self.nt_model_builder.joint_limit_upper, device=self.device).float()
        self.joint_means = (self.joint_lowers + self.joint_uppers) / 2
        self.joint_ranges = self.joint_means - self.joint_lowers
        self.joint_names = self.nt_model_builder.joint_label
        self.n_dofs = self.nt_model_builder.joint_count  # only used here for robot hand with no mimic joint

        # Original geom meshes
        self.nt_backend.step()
        self.nt_orig_geoms_meta: dict[str, NewtonShapeMeta] = (
            nt_get_links_shapes_meta(self.nt_model_builder,
                                     # state=self.nt_backend.state_0,
                                     activated_link_names=self.hand_body_names,
                                     collision_only=self.use_collision_mesh,
                                     visualized=False))

        self.wp_geom_meshes: dict[str, wp.Mesh] = {}

        # Create cache data
        if self.regen_cache or not (self.geom_convex_meshes and
                                    self.hand_surface_points and self.visible_point_indices):
            self.create_assets()
        else:
            self.make_contact_points = False
        self.hand_segment_indices, self.hand_finger_indices = self.get_hand_segment_indices()

    def create_newton_backend(self, hand_model_desc: str, hand_base_pose: np.ndarray) -> NewtonBackend:
        nt_single_world_model_builder = newton.ModelBuilder()
        nt_single_world_model_builder.add_mjcf(hand_model_desc, xform=wp_transform_from_mj(hand_base_pose),
                                               visual_classes=LEAP.VISUAL_CLASS_NAMES,
                                               collider_classes=LEAP.COLLISION_CLASS_NAMES,
                                               floating=True,
                                               override_root_xform=True,
                                               verbose=False)
        nt_backend = NewtonBackend(joint_names=LEAP.JOINTS_NAMES, kinematics_mode=True, headless=not self.visualized)
        nt_backend.build_model(single_world_model_builder=nt_single_world_model_builder, num_worlds=self.batch_size)
        return nt_backend

    def create_assets(self):
        '''
        To create needed assets for the first running.
        Should run before first use.
        '''

        # Hand convex meshes
        for geom_name, geom_mesh_meta in self.nt_orig_geoms_meta.items():
            mesh = geom_mesh_meta.source
            if isinstance(mesh, newton.Mesh):
                new_mesh = mesh.compute_convex_hull()
                self.geom_convex_meshes[Path(geom_name).stem] = trimesh.Trimesh(vertices=new_mesh.vertices,
                                                                                faces=new_mesh.indices.reshape(-1, 3),
                                                                                vertex_normals=new_mesh.normals)

        self.save_geom_convex_meshes(self.geom_convex_meshes)
        self.save_surface_points(self.hand_surface_points, self.geom_convex_meshes)

        # SAMPLE [self.hand_composite_points] -> [self.visible_point_indices]
        hand_whole_cvx_mesh = trimesh.util.concatenate(self.geom_convex_meshes.values())
        # Sample [hand_composite_points]
        self.get_forward_vertices(torch.tensor(self.hand_base_pose, device=self.device),
                                  torch.tensor(self.joint_angles, device=self.device))
        self.sample_visible_points(hand_whole_cvx_mesh, self.hand_composite_points)
        print("Assets created!")

    def get_forward_vertices(self, hand_base_pose: torch.Tensor, hand_qpos: torch.Tensor):
        # NOTE: Make sure the model builder set [floating=True] in [add_mjcf()]
        """
        from mjmanip.newton.newton_utils import nt_set_body_transform, nt_get_body_id
        nt_set_body_transform(nt_get_body_id(self.nt_backend.model_builder, LEAP.HAND_BASE_NAME), hand_base_pose,
                              self.nt_backend.model)
        """
        self.nt_backend.set_joint_targets(torch.cat([hand_base_pose, hand_qpos], dim=-1))
        self.nt_backend.step()
        if not self.nt_backend.headless:
            self.nt_backend.render()

        # Transform meshes
        verts = []
        verts_normals = []
        for geom_name, geom_mesh_meta in self.nt_orig_geoms_meta.items():
            newton_mesh = geom_mesh_meta.source
            if not newton_mesh or not newton_mesh.mesh:
                continue

            if geom_name in self.wp_geom_meshes:
                wp_mesh = self.wp_geom_meshes[geom_name]
            else:
                wp_mesh = newton_mesh.mesh
                assert len(newton_mesh.vertices) == len(wp_mesh.points)
                # NOTE: we need to copy the mesh points and create the Mesh from those such
                # that we can retain the original mesh points (of [newton_mesh.mesh]) expressed in body-fixed coordinates
                # and calculate new point positions based on transforms.
                wp_mesh = wp.Mesh(wp.clone(wp_mesh.points), wp_mesh.indices)
                self.wp_geom_meshes[geom_name] = wp_mesh

            # Transform points from [object_model]'s mesh frame to world frame using [object_transform]
            geom_new_transf = (nt_get_body_transform(geom_mesh_meta.body_id,
                                                     state=self.nt_backend.state_0)
                               * geom_mesh_meta.wp_local_pose)

            # NOTE: Transform from the original [newton_mesh]'s [points]
            wp.launch(kernel=wp_kernel_transform_mesh_points,
                      dim=len(wp_mesh.points),
                      inputs=[newton_mesh.mesh.points,
                              geom_new_transf,
                              wp.vec3(1, 1, 1)],
                      outputs=[wp_mesh.points],
                      device=str(self.device))

            # Refit the object with its transformed points
            # NOTE: If object poses change dynamically, then need to call refit again
            wp_mesh.refit()
            mesh_verts = wp.to_torch(wp_mesh.points)
            verts.append(mesh_verts)
            if self.make_contact_points:
                self.hand_composite_points[geom_name] = mesh_verts.detach().cpu().numpy()

            normals = wp.zeros(len(wp_mesh.points), dtype=wp.vec3, device=self.device)
            wp.launch(kernel=wp_kernel_compute_vertex_normals,
                      dim=len(wp_mesh.points),
                      inputs=[wp_mesh.points, wp_mesh.indices],
                      outputs=[normals],
                      device=self.device)
            verts_normals.append(wp.to_torch(normals))

        verts = torch.concatenate(verts).unsqueeze(0).float().to(self.device)
        verts_normals = torch.concatenate(verts_normals).unsqueeze(0).float().to(self.device)
        return verts, verts_normals


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    theta = np.zeros(16, dtype=np.float32)
    hand_base_pose = np.array([0, 0, 5, 1, 0, 0, 0])
    nt_hand_layer = NTLeapHandLayer(hand_model_desc=HAND_XML_PATH,
                                    hand_base_pose=hand_base_pose,
                                    joint_angles=theta,
                                    use_collision_mesh=False,
                                    visualized=True,
                                    device=device)

    if False:
        nt_hand_layer.nt_backend.spin()
    else:
        i = 0.0
        while nt_hand_layer.nt_backend.viewer.is_running():
            i += 1.1
            nt_hand_layer.get_forward_vertices(
                hand_base_pose=wp.to_torch(
                    wp.array(wp.transform(p=wp.vec3(wp.sin(i) * 0.1, wp.cos(i) * 0.1, 3)), device=device)),
                hand_qpos=torch.as_tensor(theta, device=device))
