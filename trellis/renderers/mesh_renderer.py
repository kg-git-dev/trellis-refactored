import torch
import torch.nn.functional as F
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.mesh.shader import HardFlatShader
from easydict import EasyDict as edict
from ..representations.mesh import MeshExtractResult


def intrinsics_to_pytorch3d_projection(
    intrinsics: torch.Tensor,
    image_size: tuple,
    near: float,
    far: float,
) -> torch.Tensor:
    """
    Convert OpenCV intrinsics to PyTorch3D's camera convention.
    PyTorch3D uses screen coordinates in [-1, 1] and a different projection matrix.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    H, W = image_size
    half_pix_center = 0.5  # PyTorch3D assumes pixel centers at 0.5

    # Adjust for PyTorch3D's NDC space (normalized device coordinates)
    proj = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    proj[0, 0] = 2 * fx / W
    proj[1, 1] = 2 * fy / H
    proj[0, 2] = 2 * (cx - half_pix_center) / W - 1
    proj[1, 2] = 2 * (cy - half_pix_center) / H - 1
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj


class MeshRenderer:
    def __init__(self, rendering_options={}, device="cuda"):
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
        })
        self.rendering_options.update(rendering_options)
        self.device = device

    def render(
        self,
        mesh: MeshExtractResult,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_types=["mask", "normal", "depth"],
    ) -> edict:
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, 3, resolution, resolution), device=self.device)
            ret_dict = {
                k: default_img if k in ["normal", "normal_map", "color"] else default_img[:1]
                for k in return_types
            }
            return ret_dict

        # Convert to PyTorch3D's Meshes format
        vertices = mesh.vertices.unsqueeze(0)  # (1, V, 3)
        faces = mesh.faces.unsqueeze(0)  # (1, F, 3)
        meshes = Meshes(verts=vertices, faces=faces)

        # PyTorch3D camera setup (FoVPerspectiveCameras)
        proj_matrix = intrinsics_to_pytorch3d_projection(
            intrinsics, (resolution, resolution), near, far
        ).unsqueeze(0)
        R = extrinsics[:3, :3].unsqueeze(0)  # (1, 3, 3)
        T = extrinsics[:3, 3].unsqueeze(0)  # (1, 3)
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            znear=near,
            zfar=far,
        )

        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=resolution * ssaa,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
        )

        # Create rasterizer
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        # Render mesh
        fragments = rasterizer(meshes)
        pix_to_face = fragments.pix_to_face  # (1, H, W, 1)
        zbuf = fragments.zbuf  # (1, H, W, 1)
        bary_coords = fragments.bary_coords  # (1, H, W, 1, 3)

        # Output dictionary
        out_dict = edict()

        # Mask (1 where mesh is visible)
        if "mask" in return_types:
            out_dict.mask = (pix_to_face >= 0).float()

        # Depth (linearized)
        if "depth" in return_types:
            out_dict.depth = zbuf

        # Normals (face or vertex normals)
        if "normal" in return_types:
            verts_normals = meshes.verts_normals_packed()  # (V, 3)
            faces_normals = verts_normals[mesh.faces].mean(dim=1)  # (F, 3)
            normals = faces_normals[pix_to_face.clamp(min=0)]  # (1, H, W, 1, 3)
            normals = normals.squeeze(-2).permute(0, 3, 1, 2)  # (1, 3, H, W)
            out_dict.normal = (normals + 1) / 2  # Normalize to [0, 1]

        # Normal map (interpolated vertex normals)
        if "normal_map" in return_types and hasattr(mesh, "vertex_normals"):
            vertex_normals = mesh.vertex_normals.unsqueeze(0)  # (1, V, 3)
            normals_map = vertex_normals.repeat(1, 1, 1)  # (1, V, 3)
            out_dict.normal_map = normals_map

        # Vertex colors (if available)
        if "color" in return_types and hasattr(mesh, "vertex_attrs"):
            colors = mesh.vertex_attrs[:, :3].unsqueeze(0)  # (1, V, 3)
            out_dict.color = colors

        # Downsample if SSAA > 1
        if ssaa > 1:
            for k in out_dict.keys():
                out_dict[k] = F.interpolate(
                    out_dict[k],
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )

        return out_dict