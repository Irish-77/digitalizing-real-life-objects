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

def convert_pointcloud_to_mesh(input_file: str) -> o3d.geometry.TriangleMesh:
    """Convert a point cloud file to a triangle mesh using Poisson surface reconstruction.

    Args:
        input_file (str): Path to the input point cloud file

    Returns:
        o3d.geometry.TriangleMesh: The generated triangle mesh

    Raises:
        ValueError: If the point cloud file is empty or cannot be read
    """

    warnings.filterwarnings('ignore', category=UserWarning, module='vtkOBJReader')

    depth = 10

    pcd = o3d.io.read_point_cloud(input_file)

    # Happens if models have only predicted smaller dataset
    if pcd.is_empty():
        raise ValueError(f"Failed to read point cloud or the point cloud is empty: {input_file}")

    # Clean up the point cloud (remove outliers)
    # -> remove_statistical_outlier typically produces a cleaner set of inliers
    cl, inliers = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.5)
    pcd = pcd.select_by_index(inliers)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=.5, max_nn=100
    ))
    # Re-orient normals so they point consistently
    pcd.orient_normals_consistent_tangent_plane(k=100)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth,
        n_threads=1 # Must be one, otherwise random crashes
    )
        

    # Crop away very low-density vertices
    densities = np.asarray(densities)
    low_density_cutoff = np.quantile(densities, 0.02)  # keep top 98% by density
    vertices_to_remove = densities < low_density_cutoff
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Smooth the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    # I tried both but couldn't see a difference in the output
    # or: mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)

    return mesh

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
        choices=["mesh", "pointcloud"],
        required=True,
        help="Type of input: 'mesh' or 'pointcloud'."
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
    mesh_exts = ['.obj', '.stl', '.ply']
    valid_exts = pointcloud_exts if args.input_type == "pointcloud" else mesh_exts
    
    # Get all input files
    input_files = get_file_paths(args.input_path, valid_exts)
    if not input_files: 
        raise ValueError(f"No {args.input_type} files found in {args.input_path}")

    for input_file in input_files:
        print(f"\nProcessing {input_file}...")
        
        # Generate output paths
        mesh_path = get_output_path(input_file, os.path.join(args.output_dir, "meshes"), "_mesh.obj")
        gif_path = get_output_path(input_file, args.output_dir, "_animation.gif")

        # Process the file
        if args.input_type == "pointcloud":
            try:
                mesh = convert_pointcloud_to_mesh(input_file)
                o3d.io.write_triangle_mesh(mesh_path, mesh)
                print(f"Mesh saved to {mesh_path}")
            except ValueError as e:
                print(f"Error processing {input_file}: {e}")
                continue
        else:
            mesh_path = input_file

        # # Create the animation
        # 2) Read the mesh file
        mesh = pv.read(mesh_path)

        # 3) Create a Plotter in off-screen mode
        plotter = pv.Plotter(off_screen=True, window_size=(args.resolution))
        plotter.add_mesh(mesh, show_edges=False)

        # Fit the camera to the object once
        plotter.show(auto_close=False)
        # Set the vertical field of view (like zoom factor)
        plotter.camera.SetViewAngle(args.vertical_fov)

        # 4) Determine bounding box and camera radius
        x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
        max_extent = max(x_max - x_min, y_max - y_min, z_max - z_min)
        radius = max_extent * 1.5
        center = mesh.center  # The focal point

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
