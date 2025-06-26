from typing import *
import trimesh

from ..representations import Strivec, Gaussian, MeshExtractResult

from .bake_texture import bake_texture_and_return_mesh

def to_glb(
    app_rep: Union[Strivec, Gaussian],
    mesh: MeshExtractResult,
    simplify: float = 0.90,
    texture_size: int = 1024,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.

    Args:
        app_rep (Union[Strivec, Gaussian]): Appearance representation.
        mesh (MeshExtractResult): Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        texture_size (int): Size of the texture.
        debug (bool): Whether to print debug information.
        verbose (bool): Whether to print progress.
    """
    mesh = bake_texture_and_return_mesh(
        app_rep,
        mesh, 
        simplify,
        texture_size,
        verbose,
        debug = True,
    )

    return mesh
   


