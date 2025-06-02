from typing import *
from functools import wraps
import inspect
from numbers import Number
import numpy as np
import torch
import xatlas
from .random_utils import sphere_hammersley_sequence
from tqdm import tqdm
import pyvista as pv

# from .render_utils import get_renderer
from ..representations import Octree, Gaussian, MeshExtractResult
from PIL import Image
import trimesh

from ..renderers import OctreeRenderer, MeshRenderer, GSplatRenderer

import torch.nn.functional as Functional

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    BlendParams,
    HardPhongShader
)
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.transforms import Transform3d

import cv2
import matplotlib.pyplot as plt
import os
import csv


def suppress_traceback(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            e.__traceback__ = e.__traceback__.tb_next.tb_next
            raise
    return wrapper

def get_device(args, kwargs):
    device = None
    for arg in (list(args) + list(kwargs.values())):
        if isinstance(arg, torch.Tensor):
            if device is None:
                device = arg.device
            elif device != arg.device:
                raise ValueError("All tensors must be on the same device.")
    return device

def get_args_order(func, args, kwargs):
    """
    Get the order of the arguments of a function.
    """
    names = inspect.getfullargspec(func).args
    names_idx = {name: i for i, name in enumerate(names)}
    args_order = []
    kwargs_order = {}
    for name, arg in kwargs.items():
        if name in names:
            kwargs_order[name] = names_idx[name]
            names.remove(name)
    for i, arg in enumerate(args):
        if i < len(names):
            args_order.append(names_idx[names[i]])
    return args_order, kwargs_order

def broadcast_args(args, kwargs, args_dim, kwargs_dim):
    spatial = []
    for arg, arg_dim in zip(args + list(kwargs.values()), args_dim + list(kwargs_dim.values())):
        if isinstance(arg, torch.Tensor) and arg_dim is not None:
            arg_spatial = arg.shape[:arg.ndim-arg_dim]
            if len(arg_spatial) > len(spatial):
                spatial = [1] * (len(arg_spatial) - len(spatial)) + spatial
            for j in range(len(arg_spatial)):
                if spatial[-j] < arg_spatial[-j]:
                    if spatial[-j] == 1:
                        spatial[-j] = arg_spatial[-j]
                    else:
                        raise ValueError("Cannot broadcast arguments.")
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and args_dim[i] is not None:
            args[i] = torch.broadcast_to(arg, [*spatial, *arg.shape[arg.ndim-args_dim[i]:]])
    for key, arg in kwargs.items():
        if isinstance(arg, torch.Tensor) and kwargs_dim[key] is not None:
            kwargs[key] = torch.broadcast_to(arg, [*spatial, *arg.shape[arg.ndim-kwargs_dim[key]:]])
    return args, kwargs, spatial

@suppress_traceback
def batched(*dims):
    """
    Decorator that allows a function to be called with batched arguments.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, device=torch.device('cpu'), **kwargs):
            args = list(args)
            # get arguments dimensions
            args_order, kwargs_order = get_args_order(func, args, kwargs)
            args_dim = [dims[i] for i in args_order]
            kwargs_dim = {key: dims[i] for key, i in kwargs_order.items()}
            # convert to torch tensor
            device = get_device(args, kwargs) or device
            for i, arg in enumerate(args):
                if isinstance(arg, (Number, list, tuple)) and args_dim[i] is not None:
                    args[i] = torch.tensor(arg, device=device)
            for key, arg in kwargs.items():
                if isinstance(arg, (Number, list, tuple)) and kwargs_dim[key] is not None:
                    kwargs[key] = torch.tensor(arg, device=device)
            # broadcast arguments
            args, kwargs, spatial = broadcast_args(args, kwargs, args_dim, kwargs_dim)
            for i, (arg, arg_dim) in enumerate(zip(args, args_dim)):
                if isinstance(arg, torch.Tensor) and arg_dim is not None:
                    args[i] = arg.reshape([-1, *arg.shape[arg.ndim-arg_dim:]])
            for key, arg in kwargs.items():
                if isinstance(arg, torch.Tensor) and kwargs_dim[key] is not None:
                    kwargs[key] = arg.reshape([-1, *arg.shape[arg.ndim-kwargs_dim[key]:]])
            # call function
            results = func(*args, **kwargs)
            type_results = type(results)
            results = list(results) if isinstance(results, (tuple, list)) else [results]
            # restore spatial dimensions
            for i, result in enumerate(results):
                results[i] = result.reshape([*spatial, *result.shape[1:]])
            if type_results == tuple:
                results = tuple(results)
            elif type_results == list:
                results = list(results)
            else:
                results = results[0]
            return results
        return wrapper
    return decorator

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

@batched(2,0,0)
def intrinsics_to_perspective(
        intrinsics: torch.Tensor,
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor],
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
        near (float | torch.Tensor): [...] near plane to clip
        far (float | torch.Tensor): [...] far plane to clip
    Returns:
        (torch.Tensor): [..., 4, 4] OpenGL perspective matrix
    """
    N = intrinsics.shape[0]
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    ret = torch.zeros((N, 4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[:, 0, 0] = 2 * fx
    ret[:, 1, 1] = 2 * fy
    ret[:, 0, 2] = -2 * cx + 1
    ret[:, 1, 2] = 2 * cy - 1
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2. * near * far / (near - far)
    ret[:, 3, 2] = -1.
    return ret

@batched(1, 1, 1)
def extrinsics_look_at(
    eye: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor
) -> torch.Tensor:
    """
    Get OpenCV extrinsics matrix looking at something

    Args:
        eye (torch.Tensor): [..., 3] the eye position
        look_at (torch.Tensor): [..., 3] the position to look at
        up (torch.Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (torch.Tensor): [..., 4, 4], extrinsics matrix
    """
    N = eye.shape[0]
    z = look_at - eye
    x = torch.cross(-up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    # x = torch.cross(y, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    R = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(R, eye[..., None])
    ret = torch.zeros((N, 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = R
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.
    return ret

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

def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics

@batched(0,0,0,0)
def intrinsics_from_focal_center(
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix

    Args:
        focal_x (float | torch.Tensor): focal length in x axis
        focal_y (float | torch.Tensor): focal length in y axis
        cx (float | torch.Tensor): principal point in x axis
        cy (float | torch.Tensor): principal point in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    N = fx.shape[0]
    ret = torch.zeros((N, 3, 3), dtype=fx.dtype, device=fx.device)
    zeros, ones = torch.zeros(N, dtype=fx.dtype, device=fx.device), torch.ones(N, dtype=fx.dtype, device=fx.device)
    ret = torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1).unflatten(-1, (3, 3))
    return ret

def intrinsics_from_fov_xy(
    fov_x: Union[float, torch.Tensor],
    fov_y: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix from field of view in x and y axis

    Args:
        fov_x (float | torch.Tensor): field of view in x axis
        fov_y (float | torch.Tensor): field of view in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    focal_x = 0.5 / torch.tan(fov_x / 2)
    focal_y = 0.5 / torch.tan(fov_y / 2)
    cx = cy = 0.5
    return intrinsics_from_focal_center(focal_x, focal_y, cx, cy)

def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, gs_renderer='gsplat', **kwargs):
    if isinstance(sample, Octree):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        if gs_renderer == 'gsplat':
            renderer = GSplatRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 1)
        renderer.rendering_options.far = options.get('far', 100)
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if not isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
        else:
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets: rets['normal'] = []
            rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
    return rets

def render_multiview(sample, resolution=512, nviews=30, r=2, fov=40):
    # r = 2
    # fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics

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
    near: float = 0.1,
    far: float = 10.0,
    debug: bool = True,
    verbose: bool = True,
):
    """
    Same bake as before but with extra per-view diagnostics:
      • pixel counts for rasteriser, mask, and their overlap
      • vertices in camera space (min/max Z)
      • first-view wireframe overlay
    """

    debug = False
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