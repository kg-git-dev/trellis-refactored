
import torch

from .random_utils import sphere_hammersley_sequence
import utils3d
from tqdm import tqdm
import igraph
import numpy as np
from pymeshfix import _meshfix

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt

@torch.no_grad()
def _fill_holes(
    verts,
    faces,
    max_hole_size=0.04,
    max_hole_nbe=55,
    resolution=1024,
    num_views=10,
    debug=True,
    verbose=True
):
    """
    Rasterize a mesh from multiple views and remove invisible faces.
    Also includes postprocessing to:
        1. Remove connected components that are have low visibility.
        2. Mincut to remove faces at the inner side of the mesh connected to the outer side with a small hole.

    Args:
        verts (torch.Tensor): Vertices of the mesh. Shape (V, 3).
        faces (torch.Tensor): Faces of the mesh. Shape (F, 3).
        max_hole_size (float): Maximum area of a hole to fill.
        resolution (int): Resolution of the rasterization.
        num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """
    # Construct cameras
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws = torch.tensor(yaws).cuda()
    pitchs = torch.tensor(pitchs).cuda()
    radius = 2.0
    fov = torch.deg2rad(torch.tensor(40)).cuda()
    views = []
    for (yaw, pitch) in zip(yaws, pitchs):
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda().float() * radius
        view = utils3d.torch.view_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        views.append(view)
    views = torch.stack(views, dim=0)

    # Rasterize
    device = verts.device
    fov_deg = 40.0
    radius  = 2.0
    
    # yaws, pitchs are already float32 on GPU
    dirs = torch.stack([
        torch.sin(yaws) * torch.cos(pitchs),   # x
        torch.cos(yaws) * torch.cos(pitchs),   # y
        torch.sin(pitchs)                      # z
    ], dim=1)                                  # (N, 3)

    center   = verts.mean(0)                       # (3,) float32
    verts_c  = verts - center                      # shift mesh to origin
    # recompute bounding radius
    radius   = verts_c.norm(dim=1).max() * 1.3     # 30 % margin

    # Hammersley directions → eye positions
    eyes = dirs * radius + center                  # (N,3)
    
    dtype = torch.get_default_dtype()      # float64 

    eyes   = eyes.to(dtype)
    center = center.to(dtype)
    at     = center.expand_as(eyes)
    up     = torch.tensor([0., 0., 1.], device=verts.device, dtype=dtype
                        ).expand_as(eyes)

    R, T = look_at_view_transform(eye=eyes, at=at, up=up, device=verts.device)
    
    verts_cam = (R[0] @ verts.T + T[0][:, None]).T
    z_min, z_max = verts_cam[:,2].min(), verts_cam[:,2].max()
    print(f"[DBG] view 0 Z range: {z_min.item():.2f} .. {z_max.item():.2f}")

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov_deg)

    rast_settings = RasterizationSettings(
        image_size       = resolution,
        faces_per_pixel  = 4,          # small but catches thin slivers
        blur_radius      = 0.0,
        perspective_correct = False,
        cull_backfaces   = True,       # critical for hole detection
    )

    mesh = Meshes(verts=[verts], faces=[faces.long()])
    rasterizer = MeshRasterizer(raster_settings=rast_settings)

    visibility = torch.zeros(faces.shape[0], dtype=torch.int32, device=device)
    for i in tqdm(range(num_views), disable=not verbose, desc="Rasterising"):
        fragments   = rasterizer(mesh, cameras=cameras[i])
        if debug and i in {0, int(num_views//4), int(num_views//2)}:
            mask = (fragments.pix_to_face[0,...,0] >= 0).float()
            img = (mask.cpu().numpy() * 255).astype(np.uint8)
            from PIL import Image
            Image.fromarray(img).save(f"dbg_silhouette_{i:03}.png")
        if debug and i == 0:
            zbuf = fragments.zbuf[0]
            valid_z = zbuf[zbuf < 1e10]
            print(f"[DBG] view {i}: z-range {valid_z.min():.3f} .. {valid_z.max():.3f}")
        pix2face    = fragments.pix_to_face[0, ..., 0]
        face_ids    = torch.unique(pix2face[pix2face >= 0]).long()
        visibility[face_ids] += 1

    visibility = visibility.float() / num_views

    num_hidden = (visibility == 0).sum().item()
    print(f"[DBG] faces with vis == 0 : {num_hidden} / {faces.shape[0]}")

    if debug:
        import matplotlib.pyplot as plt

        vis_color = plt.cm.viridis(visibility.cpu().numpy())[:, :3]  # RGB
        face_centers = verts[faces].mean(dim=1).cpu().numpy()
        utils3d.io.write_ply("dbg_visibility.ply", face_centers,
                            vertex_colors=(vis_color * 255).astype(np.uint8))

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
        visibility_median = visibility[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f'visibility_median: {visibility_median}')
        if visibility_median > 0.25:
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
        
        # vis_verts = verts.cpu().numpy()
        # vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
        # utils3d.io.write_ply('dbg_cut.ply', vis_verts, edges=vis_edges)
        
    
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

    if debug:
        utils3d.io.write_ply("dbg_mesh_original.ply", verts.cpu().numpy(), faces.cpu().numpy())
        utils3d.io.write_ply("dbg_mesh_filled.ply", verts.cpu().numpy(), faces.cpu().numpy())

    if debug:
        vis_color = plt.cm.viridis(visibility.cpu().numpy())[:, :3]  # RGB
        v_colors = np.zeros((verts.shape[0], 3), dtype=np.uint8)
        face_color = (vis_color * 255).astype(np.uint8)
        for fidx, face in enumerate(faces.cpu().numpy()):
            v_colors[face] = face_color[fidx]  # not accurate but good enough
        utils3d.io.write_ply("dbg_mesh_viscolor.ply", verts.cpu().numpy(), faces.cpu().numpy(), vertex_colors=v_colors)

    return verts, faces