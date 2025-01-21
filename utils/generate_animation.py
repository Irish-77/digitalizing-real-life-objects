#!/usr/bin/env python3

"""
3D Model Animation Generator

This module provides functionality to create animated GIFs from 3D models,
supporting both mesh and point cloud inputs. It handles model loading,
point cloud to mesh conversion, and creates rotating animations with
customizable parameters.

Main features:
- Point cloud to mesh conversion
- Mesh loading and visualization
- Animated GIF generation with configurable rotation patterns
"""

from typing import Tuple, List, Optional
from pathlib import Path
import pyvista as pv
import numpy as np
import open3d as o3d
import os
import argparse
import glob
import warnings
import vtk
import trimesh
import pyrender

import sys
import io

vtk.vtkObject.GlobalWarningDisplayOff()


def euler_matrix_z_y_x(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Compute a combined rotation matrix for rotations about Z, Y, and X axes.

    Args:
        alpha (float): Rotation angle about Z-axis in radians
        beta (float): Rotation angle about Y-axis in radians
        gamma (float): Rotation angle about X-axis in radians

    Returns:
        np.ndarray: 3x3 rotation matrix combining all three rotations
    """
    # Rotation about Z-axis by alpha
    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha),  0],
        [np.sin(alpha),  np.cos(alpha),  0],
        [0,              0,             1]
    ])
    
    # Rotation about Y-axis by beta
    Ry = np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [ 0,            1, 0           ],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    
    # Rotation about X-axis by gamma
    Rx = np.array([
        [1, 0,             0            ],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma),  np.cos(gamma)]
    ])
    
    # Combined rotation = Rz * Ry * Rx
    return Rz @ Ry @ Rx

def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_file_paths(input_path: str, extensions: List[str]) -> List[str]:
    """Get all files with given extensions from a directory or single file.
    
    Args:
        input_path: Path to file or directory
        extensions: List of file extensions to match (e.g., ['.txt', '.obj'])
    
    Returns:
        List of file paths
    """
    if os.path.isfile(input_path):
        return [input_path]
    
    path = Path(input_path)
    files = []
    for ext in extensions:
        files.extend(glob.glob(str(path / f"*{ext}")))
    return sorted(files)

def get_output_path(input_file: str, output_dir: str, suffix: str) -> str:
    """Generate output path based on input file and output directory."""
    ensure_directory(output_dir)
    base_name = Path(input_file).stem
    return str(Path(output_dir) / f"{base_name}{suffix}")

def convert_trimesh_to_pyvista(trimesh_mesh):
    """Convert a trimesh mesh to a PyVista mesh, preserving colors and textures."""
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces
    
    # PyVista needs faces with count prepended
    faces_with_count = np.column_stack((np.full(len(faces), 3), faces))
    faces_with_count = faces_with_count.flatten()
    
    mesh = pv.PolyData(vertices, faces_with_count)
    
    # Handle vertex colors if they exist
    if hasattr(trimesh_mesh.visual, 'vertex_colors'):
        vertex_colors = trimesh_mesh.visual.vertex_colors
        if vertex_colors is not None:
            mesh.point_data['RGB'] = vertex_colors[:, :3]
    
    # Handle face colors if they exist
    if hasattr(trimesh_mesh.visual, 'face_colors'):
        face_colors = trimesh_mesh.visual.face_colors
        if face_colors is not None:
            mesh.cell_data['RGB'] = face_colors[:, :3]
            
    return mesh

def convert_pyrender_to_pyvista(pyrender_mesh):
    """Convert a pyrender mesh to a PyVista mesh."""
    vertices = pyrender_mesh.primitives[0].positions
    indices = pyrender_mesh.primitives[0].indices
    
    # PyVista needs faces with count prepended
    faces_with_count = np.column_stack((np.full(len(indices), 3), indices))
    faces_with_count = faces_with_count.flatten()
    
    mesh = pv.PolyData(vertices, faces_with_count)
    
    # Handle colors if available
    if pyrender_mesh.primitives[0].color_0 is not None:
        mesh.point_data['RGB'] = pyrender_mesh.primitives[0].color_0[:, :3]
    
    return mesh

def load_glb_file(file_path):
    """Load a GLB file and convert it to a PyVista mesh."""
    # Load the GLB file with trimesh
    trimesh_scene = trimesh.load(file_path)
    
    # Create a pyrender scene
    scene = pyrender.Scene()
    camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 2.0],
    [0.0, 0.0, 0.0, 1.0]
    ])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)
    
    # Convert all meshes from the trimesh scene to pyrender
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for mesh in trimesh_scene.geometry.values():
        if isinstance(mesh, trimesh.Trimesh):
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
            all_vertices.append(mesh.vertices)
            all_faces.append(mesh.faces + vertex_offset)
            vertex_offset += len(mesh.vertices)
    
    # Combine all meshes into one
    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    
    # Create PyVista mesh
    faces_with_count = np.column_stack((np.full(len(faces), 3), faces))
    faces_with_count = faces_with_count.flatten()
    
    return pv.PolyData(vertices, faces_with_count)


def main() -> None:
    """Main function to process command line arguments and generate the animation.

    Command line arguments:
        --input_type: Type of input file ('mesh' or 'pointcloud')
        --input_file: Path to the input file
        --output_mesh: Path for the converted mesh (if input is pointcloud)
        --output_gif: Path for the output animation
        --resolution: Output resolution as WIDTH HEIGHT
        --frames: Number of frames in the animation
        --vertical_fov: Vertical field of view in degrees
        --speed_factors: Rotation speed multipliers for each axis

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Process single files or directories of 3D models and create animations."
    )
    parser.add_argument(
        "--input_type",
        choices=["mesh", "glb"],
        required=True,
        help="Type of input: 'mesh', or 'glb'. 'pointcloud' is deprecated."
    )
    parser.add_argument(
        "--input_path",
        "-i",
        required=True,
        help="Path to input file or directory."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        required=True,
        help="Directory for output files."
    )
    parser.add_argument(
        "--resolution",
        "-r",
        nargs=2,
        type=int,
        default=[500, 300],
        metavar=("WIDTH", "HEIGHT"),
        help="Resolution of the output frames, e.g., --resolution 1280 720."
    )
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=120,
        help="Number of frames in the GIF."
    )
    parser.add_argument(
        "--vertical_fov",
        type=float,
        default=50.0,
        help="Vertical field of view in degrees (default: 30.0)."
    )
    parser.add_argument(
        "--speed_factors",
        nargs=3,
        type=float,
        default=[1.0, 2.0, 3.0],
        metavar=("FA", "FB", "FC"),
        help="Multipliers for Euler angles (alpha, beta, gamma). Defaults to 2,4,6."
    )
    args = parser.parse_args()

    ensure_directory(args.output_dir)
    
    # Define valid extensions
    pointcloud_exts = ['.txt', '.xyz', '.pcd', '.ply']
    mesh_exts = ['.obj', '.stl', '.ply', '.glb']
    valid_exts = pointcloud_exts if args.input_type == "pointcloud" else mesh_exts
    
    # Get all input files
    input_files = get_file_paths(args.input_path, valid_exts)
    if not input_files: 
        raise ValueError(f"No {args.input_type} files found in {args.input_path}")

    for input_file in input_files:
        print(f"\nProcessing {input_file}...")
        
        # Generate output paths
        gif_path = get_output_path(input_file, args.output_dir, "_animation.gif")

        # Process the file
        if args.input_type == "mesh":
            mesh_path = input_file
            scene_or_mesh = trimesh.load(mesh_path)
            
            if isinstance(scene_or_mesh, trimesh.Scene):
                print("Loaded file is a scene. Extracting geometry...")
                mesh = trimesh.util.concatenate([geometry for geometry in scene_or_mesh.geometry.values()])
            else:
                print("Loaded file is a single mesh.")
                mesh = scene_or_mesh
                
            # Convert trimesh to PyVista mesh
            pv_mesh = convert_trimesh_to_pyvista(mesh)
            
        elif args.input_type == "glb":
            print("Loading GLB file...")
            pv_mesh = load_glb_file(input_file)

            # Set up the plotter with the correct color mapping
            plotter = pv.Plotter(off_screen=True, window_size=(args.resolution))
            plotter.add_mesh(
                pv_mesh,
                rgb=True,
                show_edges=False,
                interpolate_before_map=True
            )



        # Continue with existing animation code...
        plotter = pv.Plotter(off_screen=True, window_size=(args.resolution))
        if 'RGB' in pv_mesh.point_data:
            plotter.add_mesh(pv_mesh, scalars='RGB', rgb=True, show_edges=False)
        elif 'RGB' in pv_mesh.cell_data:
            plotter.add_mesh(pv_mesh, scalars='RGB', rgb=True, show_edges=False)
        else:
            plotter.add_mesh(pv_mesh, show_edges=False)

        # Fit the camera to the object once
        plotter.show(auto_close=False)
        # Set the vertical field of view (like zoom factor)
        plotter.camera.SetViewAngle(args.vertical_fov)

        # 4) Determine bounding box and camera radius
        x_min, x_max, y_min, y_max, z_min, z_max = pv_mesh.bounds
        max_extent = max(x_max - x_min, y_max - y_min, z_max - z_min)
        radius = max_extent * 1.5
        center = pv_mesh.center  # The focal point

        # 5) Open GIF for writing frames
        plotter.open_gif(gif_path)

        # Ensure the animation ends where it starts (i.e., loops seamlessly):
        for i in range(args.frames):
            if args.frames > 1:
                t = i / (args.frames - 1)
            else:
                t = 0

            # Euler angles in radians; each goes 0->2 \pi * factor
            alpha = 2.0 * np.pi * args.speed_factors[0] * t
            beta  = 2.0 * np.pi * args.speed_factors[1] * t
            gamma = 2.0 * np.pi * args.speed_factors[2] * t

            # Compute rotation matrix
            R = euler_matrix_z_y_x(alpha, beta, gamma)

            # Baseline camera vector (pointing along +Z by default)
            baseline_cam_pos = np.array([0.0, 0.0, radius])
            rotated_cam_pos = R @ baseline_cam_pos

            # Also rotate the "up" vector
            baseline_up = np.array([0.0, 1.0, 0.0])
            rotated_up = R @ baseline_up

            # Set the camera position
            plotter.camera_position = [
                (rotated_cam_pos[0] + center[0],
                    rotated_cam_pos[1] + center[1],
                    rotated_cam_pos[2] + center[2]),
                center,
                (rotated_up[0], rotated_up[1], rotated_up[2])
            ]

            plotter.render()
            plotter.write_frame()

        plotter.close()
        print(f"Animation saved to {gif_path}")

if __name__ == "__main__":
    main()
