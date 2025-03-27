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
            "resolution": 512,
            "near": 1.0,
            "far": 100.0,
            "ssaa": 1,
            "bg_color": (0, 0, 0)
        })
        self.rendering_options.update(rendering_options)
        self.device = device

    def render(
        self,
        mesh: MeshExtractResult,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_types=["normal"],
    ) -> edict:
        resolution = self.rendering_options["resolution"]
        ssaa = self.rendering_options["ssaa"]
        
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return {"normal": torch.zeros((3, resolution, resolution), device=self.device)}

        # Prepare mesh data
        vertices = mesh.vertices.unsqueeze(0).to(self.device)  # (1, V, 3)
        faces = mesh.faces.unsqueeze(0).to(self.device)  # (1, F, 3)
        meshes = Meshes(verts=vertices, faces=faces)

        # Camera setup
        R = extrinsics[:3, :3].unsqueeze(0).to(self.device)
        T = extrinsics[:3, 3].unsqueeze(0).to(self.device)
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            znear=self.rendering_options["near"],
            zfar=self.rendering_options["far"],
        )

        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=resolution * ssaa,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Render mesh
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )
        fragments = rasterizer(meshes)

        # Process normals
        verts_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = verts_normals[mesh.faces].mean(dim=1)  # (F, 3)
        visible_faces = fragments.pix_to_face.clamp(min=0)  # (1, H, W, 1)
        
        # Get normals for visible pixels
        normals = faces_normals[visible_faces.view(-1)]  # (N, 3)
        normals = normals.view(1, resolution*ssaa, resolution*ssaa, 3)  # (1, H, W, 3)
        
        # Normalize to [0,1] and reshape to (1, 3, H, W)
        normals = ((normals + 1) / 2).permute(0, 3, 1, 2)
        
        # Downsample if needed
        if ssaa > 1:
            normals = F.interpolate(
                normals,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )

        return {"normal": normals.squeeze(0)}  # (3, H, W)