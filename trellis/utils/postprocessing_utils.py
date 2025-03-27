from typing import *
import numpy as np
import torch
import utils3d
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    TexturesUV,
)
from pytorch3d.structures import Meshes
from tqdm import tqdm
import trimesh
import trimesh.visual
import xatlas
import pyvista as pv
from pymeshfix import _meshfix
import igraph
import cv2
from PIL import Image
from .random_utils import sphere_hammersley_sequence
from .render_utils import render_multiview
from ..representations import Strivec, Gaussian, MeshExtractResult


@torch.no_grad()
def _fill_holes(
    verts: torch.Tensor,
    faces: torch.Tensor,
    max_hole_size: float = 0.04,
    max_hole_nbe: int = 32,
    resolution: int = 128,
    num_views: int = 500,
    debug: bool = False,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize a mesh from multiple views and remove invisible faces (PyTorch3D version).
    """
    device = verts.device

    # Construct cameras (PyTorch3D expects y-down coordinate system)
    yaws, pitchs = [], []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws = torch.tensor(yaws, device=device)
    pitchs = torch.tensor(pitchs, device=device)
    radius = 2.0
    fov = torch.deg2rad(torch.tensor(40, device=device))

    # PyTorch3D camera setup
    cameras = []
    for yaw, pitch in zip(yaws, pitchs):
        origin = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ], device=device) * radius
        R = utils3d.torch.look_at_rotation(origin, torch.zeros(3, device=device))
        T = -R @ origin.unsqueeze(-1)
        cameras.append(FoVPerspectiveCameras(
            device=device,
            R=R.T.unsqueeze(0),  # PyTorch3D expects R.T
            T=T.T.unsqueeze(0),
            fov=fov,
        ))

    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Initialize visibility counter
    visibility = torch.zeros(faces.shape[0], dtype=torch.int32, device=device)

    # Rasterize from multiple views
    meshes = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0))
    for i, cam in enumerate(tqdm(cameras, disable=not verbose, desc="Rasterizing")):
        rasterizer = MeshRasterizer(cameras=cam, raster_settings=raster_settings)
        fragments = rasterizer(meshes)
        visible_faces = fragments.pix_to_face.unique()  # Get visible face indices
        visible_faces = visible_faces[visible_faces >= 0]  # Filter background
        visibility[visible_faces] += 1

    visibility = visibility.float() / num_views
    
    # Mincut
    ## construct outer faces
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    connected_components = utils3d.torch.compute_connected_components(faces, edges, face2edge)
    outer_face_indices = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    for i in range(len(connected_components)):
        outer_face_indices[connected_components[i]] = visibility[connected_components[i]] > min(max(visibility[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)
    
    ## construct inner faces
    inner_face_indices = torch.nonzero(visibility == 0).reshape(-1)
    if verbose:
        tqdm.write(f'Found {inner_face_indices.shape[0]} invisible faces')
    if inner_face_indices.shape[0] == 0:
        return verts, faces
    
    ## Construct dual graph (faces as nodes, edges as edges)
    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_edges_weights = torch.norm(verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1)
    if verbose:
        tqdm.write(f'Dual graph: {dual_edges.shape[0]} edges')

    ## solve mincut problem
    ### construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])
    g.add_edges(dual_edges.cpu().numpy())
    g.es['weight'] = dual_edges_weights.cpu().numpy()
    
    ### source and target
    g.add_vertex('s')
    g.add_vertex('t')
    
    ### connect invisible faces to source
    g.add_edges([(f, 's') for f in inner_face_indices], attributes={'weight': torch.ones(inner_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})
    
    ### connect outer faces to target
    g.add_edges([(f, 't') for f in outer_face_indices], attributes={'weight': torch.ones(outer_face_indices.shape[0], dtype=torch.float32).cpu().numpy()})
                
    ### solve mincut
    cut = g.mincut('s', 't', (np.array(g.es['weight']) * 1000).tolist())
    remove_face_indices = torch.tensor([v for v in cut.partition[0] if v < faces.shape[0]], dtype=torch.long, device=faces.device)
    if verbose:
        tqdm.write(f'Mincut solved, start checking the cut')
    
    ### check if the cut is valid with each connected component
    to_remove_cc = utils3d.torch.compute_connected_components(faces[remove_face_indices])
    if debug:
        tqdm.write(f'Number of connected components of the cut: {len(to_remove_cc)}')
    valid_remove_cc = []
    cutting_edges = []
    for cc in to_remove_cc:
        #### check if the connected component has low visibility
        visblity_median = visibility[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f'visblity_median: {visblity_median}')
        if visblity_median > 0.25:
            continue
        
        #### check if the cuting loop is small enough
        cc_edge_indices, cc_edges_degree = torch.unique(face2edge[remove_face_indices[cc]], return_counts=True)
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)]
        if len(cc_new_boundary_edge_indices) > 0:
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(edges[cc_new_boundary_edge_indices])
            cc_new_boundary_edges_cc_center = [verts[edges[cc_new_boundary_edge_indices[edge_cc]]].mean(dim=1).mean(dim=0) for edge_cc in cc_new_boundary_edge_cc]
            cc_new_boundary_edges_cc_area = []
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]] - cc_new_boundary_edges_cc_center[i]
                _e2 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]] - cc_new_boundary_edges_cc_center[i]
                cc_new_boundary_edges_cc_area.append(torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5)
            if debug:
                cutting_edges.append(cc_new_boundary_edge_indices)
                tqdm.write(f'Area of the cutting loop: {cc_new_boundary_edges_cc_area}')
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue
            
        valid_remove_cc.append(cc)
        
    if debug:
        face_v = verts[faces].mean(dim=1).cpu().numpy()
        vis_dual_edges = dual_edges.cpu().numpy()
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]
        if len(valid_remove_cc) > 0:
            vis_colors[remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()] = [255, 0, 0]
        utils3d.io.write_ply('dbg_dual.ply', face_v, edges=vis_dual_edges, vertex_colors=vis_colors)
        
        vis_verts = verts.cpu().numpy()
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
        utils3d.io.write_ply('dbg_cut.ply', vis_verts, edges=vis_edges)
        
    
    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
        if verbose:
            tqdm.write(f'Removed {(~mask).sum()} faces by mincut')
    else:
        if verbose:
            tqdm.write(f'Removed 0 faces by mincut')
            
    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
    verts, faces = torch.tensor(verts, device='cuda', dtype=torch.float32), torch.tensor(faces, device='cuda', dtype=torch.int32)

    return verts, faces


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    fill_holes: bool = True,
    fill_holes_max_hole_size: float = 0.04,
    fill_holes_max_hole_nbe: int = 32,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    debug: bool = False,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_hole_size (float): Maximum area of a hole to fill.
        fill_holes_max_hole_nbe (int): Maximum number of boundary edges of a hole to fill.
        fill_holes_resolution (int): Resolution of the rasterization.
        fill_holes_num_views (int): Number of views to rasterize the mesh.
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

    # Remove invisible faces
    if fill_holes:
        vertices, faces = torch.tensor(vertices).cuda(), torch.tensor(faces.astype(np.int32)).cuda()
        vertices, faces = _fill_holes(
            vertices, faces,
            max_hole_size=fill_holes_max_hole_size,
            max_hole_nbe=fill_holes_max_hole_nbe,
            resolution=fill_holes_resolution,
            num_views=fill_holes_num_views,
            debug=debug,
            verbose=verbose,
        )
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
        if verbose:
            tqdm.write(f'After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    return vertices, faces


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


def bake_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    observations: List[np.ndarray],
    masks: List[np.ndarray],
    extrinsics: List[np.ndarray],
    intrinsics: List[np.ndarray],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal["fast", "opt"] = "opt",
    lambda_tv: float = 1e-2,
    verbose: bool = False,
) -> np.ndarray:
    """
    Bake texture using PyTorch3D (replaces NVDiffrast-based baking).
    """
    device = "cuda"
    vertices = torch.tensor(vertices, device=device)
    faces = torch.tensor(faces.astype(np.int32), device=device)
    uvs = torch.tensor(uvs, device=device)
    textures = torch.zeros((1, texture_size, texture_size, 3), device=device)

    # Convert observations to PyTorch tensors
    obs_tensors = [torch.tensor(obs / 255.0, device=device) for obs in observations]
    mask_tensors = [torch.tensor(m > 0, device=device) for m in masks]

    # Create PyTorch3D cameras
    cameras = []
    for extr, intr in zip(extrinsics, intrinsics):
        R = torch.tensor(extr[:3, :3], device=device).unsqueeze(0)
        T = torch.tensor(extr[:3, 3], device=device).unsqueeze(0)
        cameras.append(FoVPerspectiveCameras(
            device=device,
            R=R,
            T=T,
            znear=near,
            zfar=far,
            fov=2 * torch.atan2(intr[1, 2], intr[1, 1]),
        ))

    if mode == "fast":
        # Fast mode: Project observations to UV space
        for cam, obs, mask in zip(cameras, obs_tensors, mask_tensors):
            meshes = Meshes(
                verts=vertices.unsqueeze(0),
                faces=faces.unsqueeze(0),
                textures=TexturesUV(
                    maps=textures,
                    faces_uvs=faces.unsqueeze(0),
                    verts_uvs=uvs.unsqueeze(0),
                ),
            )
            rasterizer = MeshRasterizer(
                cameras=cam,
                raster_settings=RasterizationSettings(
                    image_size=obs.shape[:2],
                    blur_radius=0.0,
                    faces_per_pixel=1,
                ),
            )
            fragments = rasterizer(meshes)
            visible_pixels = (fragments.pix_to_face >= 0) & mask.unsqueeze(-1)
            # Update texture (simplified projection)
            # ... (implement UV projection logic) ...

    elif mode == "opt":
        # Optimization-based texture baking
        textures = torch.nn.Parameter(textures)
        optimizer = torch.optim.Adam([textures], lr=1e-2)

        for step in tqdm(range(1000), disable=not verbose, desc="Texture optimization"):
            loss = 0.0
            for cam, obs, mask in zip(cameras, obs_tensors, mask_tensors):
                meshes = Meshes(
                    verts=vertices.unsqueeze(0),
                    faces=faces.unsqueeze(0),
                    textures=TexturesUV(
                        maps=textures,
                        faces_uvs=faces.unsqueeze(0),
                        verts_uvs=uvs.unsqueeze(0),
                    ),
                )
                rasterizer = MeshRasterizer(
                    cameras=cam,
                    raster_settings=RasterizationSettings(
                        image_size=obs.shape[:2],
                        blur_radius=0.0,
                        faces_per_pixel=1,
                    ),
                )
                fragments = rasterizer(meshes)
                rendered = meshes.textures.sample_textures(fragments.bary_coords)
                mask = (fragments.pix_to_face >= 0) & mask.unsqueeze(-1)
                loss += F.mse_loss(rendered[mask], obs[mask])

            # Total variation regularization
            if lambda_tv > 0:
                tv_loss = (
                    torch.sum(torch.abs(textures[:, :-1, :, :] - textures[:, 1:, :, :])) +
                    torch.sum(torch.abs(textures[:, :, :-1, :] - textures[:, :, 1:, :]))
                )
                loss += lambda_tv * tv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Post-process texture (inpainting, etc.)
    texture = textures[0].detach().cpu().numpy()
    texture = (np.clip(texture, 0, 1) * 255).astype(np.uint8)
    mask = (texture.sum(axis=-1) == 0).astype(np.uint8)
    texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    return texture


def to_glb(
    app_rep: Union[Strivec, Gaussian],
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    debug: bool = False,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.

    Args:
        app_rep (Union[Strivec, Gaussian]): Appearance representation.
        mesh (MeshExtractResult): Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_size (float): Maximum area of a hole to fill.
        texture_size (int): Size of the texture.
        debug (bool): Whether to print debug information.
        verbose (bool): Whether to print progress.
    """
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    
    # mesh postprocess
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=fill_holes_max_size,
        fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)),
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=debug,
        verbose=verbose
    )

    # parametrize mesh
    vertices, faces, uvs = parametrize_mesh(vertices, faces)

    # bake texture
    observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=1024, nviews=100)
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
    texture = bake_texture(
        vertices, faces, uvs,
        observations, masks, extrinsics, intrinsics,
        texture_size=texture_size, mode='opt',
        lambda_tv=0.01,
        verbose=verbose
    )
    texture = Image.fromarray(texture)

    # rotate mesh (from z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, image=texture))
    return mesh