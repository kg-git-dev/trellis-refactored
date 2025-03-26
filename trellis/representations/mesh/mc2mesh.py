import numpy as np
import torch
from easydict import EasyDict as edict
from skimage import measure
from typing import Tuple, Optional

from .cube2mesh import MeshExtractResult
from .utils_cube import *
from ...modules.sparse import SparseTensor
import torchmcubes

class EnhancedMarchingCubes:
    def __init__(self, device="cuda"):
        self.device = device
        self.zero = torch.tensor(0.0, device=device)
        self.one = torch.tensor(1.0, device=device)
        # Pre-compute offsets once
        self.offsets = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], device=device, dtype=torch.long)
        
    def __call__(self,
                voxelgrid_vertices: torch.Tensor,
                scalar_field: torch.Tensor,
                voxelgrid_colors: Optional[torch.Tensor] = None,
                training: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        # Validate inputs and standardize shapes upfront
        scalar_field = self._prepare_scalar_field(scalar_field)
        voxelgrid_vertices = self._prepare_voxelgrid(voxelgrid_vertices, scalar_field.shape[0])
        
        # Use torchmcubes for GPU acceleration
        vertices, triangles = torchmcubes.marching_cubes(scalar_field, 0.0)
        vertices = vertices.float()
        faces = triangles.long()
        
        # Process deformations and colors
        deformed_vertices = self._apply_deformations(vertices, voxelgrid_vertices)
        colors = self._process_colors(vertices, voxelgrid_colors, scalar_field.shape[0]) if voxelgrid_colors is not None else None
        
        # Compute loss if training
        deviation_loss = self._compute_deviation_loss(vertices, deformed_vertices) if training else self.zero
        
        return deformed_vertices, faces.flip(dims=[1]), deviation_loss, colors
    
    def _prepare_scalar_field(self, scalar_field: torch.Tensor) -> torch.Tensor:
        """Standardize scalar field shape and move to GPU"""

        if scalar_field.dim() == 1:
            grid_size = int(round(scalar_field.shape[0] ** (1/3)))
            scalar_field = scalar_field.view(grid_size, grid_size, grid_size)
        return scalar_field.squeeze().to(self.device)

    def _prepare_voxelgrid(self, voxelgrid: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Standardize voxel grid shape"""

        if voxelgrid is None:
            return None
        if voxelgrid.dim() == 2:
            return voxelgrid.view(grid_size, grid_size, grid_size, -1).to(self.device)
        return voxelgrid.to(self.device)
    
    def _apply_deformations(self, vertices: torch.Tensor, voxelgrid: torch.Tensor) -> torch.Tensor:
        """Optimized deformation application with fused operations"""
        
        # Convert vertices to grid coordinates in one operation
        grid_coords = vertices.long()
        local_coords = vertices - grid_coords.float()
        
        # Clamp coordinates in one operation
        max_coord = voxelgrid.shape[0] - 1
        grid_coords = torch.clamp(grid_coords, 0, max_coord)
        
        # Vectorized trilinear interpolation
        return self._trilinear_interpolate_v2(grid_coords, local_coords, voxelgrid)

    def _trilinear_interpolate_v2(self, grid_coords: torch.Tensor, 
                                local_coords: torch.Tensor,
                                values: torch.Tensor) -> torch.Tensor:
        """Vectorized trilinear interpolation"""
        # Unpack coordinates
        x, y, z = local_coords.unbind(1)
        x_ = torch.stack([1-x, x], dim=1)
        y_ = torch.stack([1-y, y], dim=1)
        z_ = torch.stack([1-z, z], dim=1)
        
        # Outer product for weights
        weights = (x_.unsqueeze(2) * y_.unsqueeze(1)).view(-1, 4, 1) * z_.unsqueeze(1)
        weights = weights.view(-1, 8)
        
        # Compute all neighbor coordinates in one operation
        neighbor_coords = grid_coords.unsqueeze(1) + self.offsets.unsqueeze(0)
        neighbor_coords = torch.clamp(neighbor_coords, 0, values.shape[0]-1)
        
        # Gather values from grid (handles both 3D and 4D tensors)
        if values.dim() == 4:
            neighbor_values = values[
                neighbor_coords[..., 0],
                neighbor_coords[..., 1], 
                neighbor_coords[..., 2]
            ]
        else:
            neighbor_values = values[
                neighbor_coords[..., 0],
                neighbor_coords[..., 1],
                neighbor_coords[..., 2]
            ].unsqueeze(-1)
        
        # Compute interpolation weights
        weights = torch.stack([
            (1-x)*(1-y)*(1-z), (1-x)*(1-y)*z, (1-x)*y*(1-z), (1-x)*y*z,
            x*(1-y)*(1-z), x*(1-y)*z, x*y*(1-z), x*y*z
        ], dim=1)
        
        return (weights.unsqueeze(-1) * neighbor_values).sum(dim=1)
    
    def _process_colors(self, vertices: torch.Tensor, 
                    colors: torch.Tensor,
                    grid_size: int) -> torch.Tensor:
        """Optimized color processing pipeline"""
        colors = self._prepare_voxelgrid(colors, grid_size)
            
        # Use the same optimized interpolation
        interpolated = self._trilinear_interpolate_v2(
            vertices.long(),
            vertices - vertices.long().float(),
            colors
        )     
        # Sigmoid in place for memory efficiency
        return torch.sigmoid_(interpolated)
    
    def _compute_deviation_loss(self, original: torch.Tensor, 
                            deformed: torch.Tensor) -> torch.Tensor:
        """Fused operation deviation loss"""
        return torch.mean(torch.square_(deformed - original))


class SparseFeatures2MCMesh:
    def __init__(self, device="cuda", res=128, use_color=True):
        super().__init__()
        self.device = device
        self.res = res
        self.mesh_extractor = EnhancedMarchingCubes(device=device)
        self.sdf_bias = -1.0 / res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self._calc_layout()

    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            '''
            6 channel color including normal map
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start

    def get_layout(self, feats: torch.Tensor, name: str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name][
            'shape'])

    def __call__(self, cubefeats: SparseTensor, training=False):
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats

        sdf, deform, color, weights = [self.get_layout(feats, name)
                                       for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1),
                                                     training=training)

        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res + 1, sdf_init=True)

        if self.use_color:
            sdf_d, deform_d, colors_d = (v_attrs_d[..., 0], v_attrs_d[..., 1:4],
                                         v_attrs_d[..., 4:])
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None

        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)

        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            voxelgrid_colors=colors_d,
            training=training
        )

        mesh = MeshExtractResult(vertices=vertices, faces=faces,
                                 vertex_attrs=colors, res=self.res)

        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:, :20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]

        return mesh