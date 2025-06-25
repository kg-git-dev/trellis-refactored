from typing import *

import numpy as np
import torch
import xatlas
from tqdm import tqdm
import pyvista as pv

from .render_utils import render_multiview

from PIL import Image
import trimesh

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    TexturesUV,
)

import cv2
import os
import csv

from .batched_helper import batched

@batched(2)
def extrinsics_to_view(extr):
    """
    Convert OpenCV-style world→cam matrix to PyTorch3D world→cam.
    Flip X and Y so that (+x right, +y down) -> (+x right, +y up).
    """
    flip = extr.new_tensor([[-1, 0, 0, 0],   # X ↦ −X
                            [ 0,-1, 0, 0],   # Y ↦ −Y
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1]])
    return flip @ extr

def fill_texture_holes(
    texture_map_tensor: torch.Tensor,
    threshold: int = 0,                 
    method: str = "telea",
    debug: bool = False                 
) -> torch.Tensor:
    """
    In-paints RGB texture maps.
    Any texel whose three channels are ≤ `threshold`
    is treated as a hole.

    Args
    ----
    texture_map_tensor : (H, W, 3) float32 ∈ [0,1]
    threshold          : 0-255 tolerance for “empty” texels
    method             : 'telea' | 'ns'  (OpenCV algorithms)
    debug              : if True print hole statistics
    """
    if texture_map_tensor.shape[-1] != 3:
        raise ValueError("Expected an RGB tensor with 3 channels")

    device = texture_map_tensor.device
    tex_np = (texture_map_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # ----------------------------- tolerant mask
    mask = (tex_np <= threshold).all(axis=-1).astype(np.uint8)  # uint8 0/1

    if debug:
        hole_ratio = mask.mean() * 100
        print(f"[DEBUG] in-paint: {hole_ratio:.2f}% of texels are holes "
              f"(threshold={threshold})")

    if not mask.any():
        return texture_map_tensor  # nothing to do

    flags = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    inpainted_rgb = cv2.inpaint(tex_np, mask, inpaintRadius=3, flags=flags)

    return (torch.from_numpy(inpainted_rgb).float() / 255.0).to(device)

def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs

def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(f'Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Simplify
    if simplify and simplify_ratio > 0:
        mesh = pv.PolyData(vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1))
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose:
            tqdm.write(f'After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')
        if verbose:
            tqdm.write(f'After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    return vertices, faces

def bake_texture_and_return_mesh(
    app_rep,
    mesh,
    simplify: float = 0.90,
    texture_size: int = 1024,
    verbose: bool = False,
    debug: bool = False,
):
    """
    Same bake as before but with extra per-view diagnostics:
      • pixel counts for rasteriser, mask, and their overlap
      • vertices in camera space (min/max Z)
      • first-view wireframe overlay
    """

    # ---------- 0. debug dirs ----------
    if debug:
        dbg = "debug_bake"
        os.makedirs(dbg, exist_ok=True)
        log_csv = open(os.path.join(dbg, "view_stats.csv"), "w", newline="")
        log = csv.writer(log_csv)
        log.writerow(["view", "rast_px", "mask_px", "ol_px",
                      "minZ", "maxZ", "medianZ"])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 1. mesh ----------
    Vn, Fn = mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy()
    Vn, Fn = postprocess_mesh(Vn, Fn, simplify_ratio=simplify)
    Vn, Fn, UVn = parametrize_mesh(Vn, Fn)

    V = torch.tensor(Vn,  dtype=torch.float32, device=dev)
    F = torch.tensor(Fn,  dtype=torch.int64,   device=dev)
    UV = torch.tensor(UVn, dtype=torch.float32, device=dev)

    faces_uvs = F

    # ---------- 2. observations ----------
    imgs, extrs, intrs = render_multiview(app_rep, 1024, 1000)
    H = W = imgs[0].shape[0]
    obs  = [torch.tensor(i/255.0, dtype=torch.float32, device=dev) for i in imgs]
    msk  = [torch.tensor((i>0).any(-1), dtype=torch.bool, device=dev) for i in imgs]

    # ---------- 3. empty-texture mesh ----------
    tex0 = torch.zeros((1, texture_size, texture_size, 3),
                       dtype=torch.float32, device=dev)
    mesh_p3d = Meshes(
        verts=[V], faces=[F],
        textures=TexturesUV(maps=tex0.permute(0,3,1,2),
                            faces_uvs=[faces_uvs], verts_uvs=[UV]))

    rast = MeshRasterizer(
        raster_settings=RasterizationSettings(image_size=(H,W),
                                              faces_per_pixel=1))

    tex_acc = torch.zeros_like(tex0[0])
    w_acc   = torch.zeros(texture_size, texture_size, 1, device=dev)

    views_ok = 0
    for k,(rgb,mask,Ex,K3) in enumerate(tqdm(zip(obs,msk,extrs,intrs),
                                             total=len(obs),
                                             disable=not verbose)):
        # ----- camera -----
        view = extrinsics_to_view(Ex.to(dev).float())
        R = view[:3, :3]                             
        T = view[:3, 3]

        verts_cam = (R @ V.T + T[:, None]).T    # V is (V,3)
        
        if debug:
            print(f"[DBG] view {k}: z range {verts_cam[:,2].min():.2f} .. {verts_cam[:,2].max():.2f}")
        
        fx,fy = K3[0,0]*W, K3[1,1]*H
        cx,cy = K3[0,2]*W, K3[1,2]*H
        cam = PerspectiveCameras(device=dev, in_ndc=False,
                                 R=R[None], T=T[None],
                                 focal_length=torch.tensor([[fx,fy]],dtype=torch.float32,device=dev),
                                 principal_point=torch.tensor([[cx,cy]],dtype=torch.float32,device=dev),
                                 image_size=torch.tensor([[H,W]],dtype=torch.int32,device=dev))

        # ----- quick Z check -----
        V_cam = (R @ V.T + T[:,None]).T       # (V,3)
        Z = V_cam[:,2]
        zmin,zmax,zmed = Z.min().item(), Z.max().item(), Z.median().item()

        # ----- rasterise -----
        frags = rast(mesh_p3d, cameras=cam)
        p2f   = frags.pix_to_face[0,...,0]
        rast_vis = p2f >= 0
        olap  = rast_vis & mask

        if debug:
            log.writerow([k, int(rast_vis.sum()), int(mask.sum()),
                          int(olap.sum()), zmin, zmax, zmed])
            if k==0:  # save first overlay even if empty
                over = np.zeros((H,W,3),np.uint8)
                over[...,2] = rast_vis.cpu().numpy()*255
                over[...,1] = mask.cpu().numpy()*255
                cv2.imwrite(f"{dbg}/overlay_{k}.png", over)

        if olap.sum()==0:
            continue  # skip this view but keep logging

        views_ok += 1
        fi = p2f[olap]; bary = frags.bary_coords[0,...,0,:][olap]
        uv = (bary[...,None]*UV[faces_uvs[fi]]).sum(-2)
        uv[...,1] = 1-uv[...,1]
        xy = (uv*(texture_size-1)).long()
        u,v = xy[:,0],xy[:,1]
        tex_acc.index_put_((v,u), rgb[olap], accumulate=True)
        w_acc .index_put_((v,u),
                          torch.ones_like(v,dtype=torch.float32)[:,None],
                          accumulate=True)

    if debug: log_csv.close()

    if views_ok==0:
        raise RuntimeError(
            "All views had zero overlap. "
            "Inspect debug_bake/overlay_0.png and view_stats.csv:\n"
            "  • If blue channel is empty → camera/extrinsics wrong.\n"
            "  • If green channel is empty → mask is empty or threshold too high."
        )

    if (w_acc>0).sum()==0:
        raise RuntimeError(
            "Rasteriser saw faces but all UVs mapped outside [0,1]. "
            "Check UV range and faces_uvs mapping."
        )

    # normalise, export etc
    tex = tex_acc / w_acc.clamp(min=1e-6)
    tex_np = (tex.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
    Image.fromarray(tex_np).save(os.path.join(dbg,"baked_texture.png")) if debug else None

    visual = trimesh.visual.TextureVisuals(
        uv=UV.cpu().numpy(),
        image=Image.fromarray(tex_np),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(tex_np)))
    up = np.array([[1, 0, 0],[0, 0, 1],[0,-1, 0]], np.float32)

    return trimesh.Trimesh((Vn@up.T), Fn, visual=visual, process=False)