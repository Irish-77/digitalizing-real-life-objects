import argparse
import pyrender
import trimesh
import numpy as np
from math import sin, cos
import time
import imageio
import os
from PIL import Image

def create_rotation_matrix(angle, axis):
    """Create rotation matrix for specific axis and angle."""
    c, s = cos(angle), sin(angle)
    if axis == 'x':
        return np.array([
            [1, 0,  0, 0],
            [0, c, -s, 0],
            [0, s,  c, 0],
            [0, 0,  0, 1]
        ])
    elif axis == 'y':
        return np.array([
            [c,  0, s, 0],
            [0,  1, 0, 0],
            [-s, 0, c, 0],
            [0,  0, 0, 1]
        ])
    else:  # z
        return np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

def get_preprocessing_steps(model_name):
    if model_name == "IM":
        FOV = np.pi / 3.4 # 3.0
        rotation_x = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        rotation_y = trimesh.transformations.rotation_matrix(-np.pi/2, [0, 0, 1])
        combined_rotation = trimesh.transformations.concatenate_matrices(rotation_x, rotation_y)
    elif model_name == "TSR":
        FOV = np.pi / 6.0
        rotation_x = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        rotation_y = trimesh.transformations.rotation_matrix(-np.pi/2, [0, 0, 1])
        rotation_z = trimesh.transformations.rotation_matrix(np.pi*1, [0, 1, 0])
        combined_rotation = trimesh.transformations.concatenate_matrices(rotation_x, rotation_y, rotation_z)
    elif model_name == "G3D":
        FOV = np.pi / 5.0
        rotation_x = trimesh.transformations.rotation_matrix(np.pi*2, [1, 0, 0])
        rotation_y = trimesh.transformations.rotation_matrix(np.pi*2, [0, 0, 1])
        rotation_z = trimesh.transformations.rotation_matrix(np.pi*2, [0, 1, 0])
        combined_rotation = trimesh.transformations.concatenate_matrices(rotation_x, rotation_y, rotation_z)
    elif model_name == "TG":
        FOV = np.pi / 4.0 #3.5
        rotation_x = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        rotation_y = trimesh.transformations.rotation_matrix(np.pi*1.5, [0, 0, 1])
        rotation_z = trimesh.transformations.rotation_matrix(np.pi*1, [0, 1, 0])
        combined_rotation = trimesh.transformations.concatenate_matrices(rotation_x, rotation_y, rotation_z)
    elif model_name == "LGM":
        FOV = np.pi / 6.0
        rotation_x = trimesh.transformations.rotation_matrix(np.pi*2, [1, 0, 0])
        rotation_y = trimesh.transformations.rotation_matrix(np.pi*2, [0, 0, 1])
        rotation_z = trimesh.transformations.rotation_matrix(np.pi*1.5, [0, 1, 0])
        combined_rotation = trimesh.transformations.concatenate_matrices(rotation_x, rotation_y, rotation_z)

    return combined_rotation, FOV


def create_animation(input_file, output_path, model_name, type, n_frames, width, height, rotation_order):
    # Load the 3D file based on format
    trimesh_scene = trimesh.load(input_file)
    if isinstance(trimesh_scene, trimesh.Trimesh):
        meshes = [trimesh_scene]
    else:
        meshes = trimesh_scene.geometry.values()

    # Get preprocessing steps based on model name
    combined_rotation, FOV = get_preprocessing_steps(model_name)

    # Apply preprocessing steps
    transformed_meshes = []
    for mesh in meshes:
        if isinstance(mesh, trimesh.Trimesh):
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_transform(combined_rotation)
            transformed_meshes.append(transformed_mesh)

    # Save transformed model
    if type == 'glb':
        # Create a new scene with transformed meshes
        transformed_scene = trimesh.Scene()
        for i, mesh in enumerate(transformed_meshes):
            transformed_scene.add_geometry(mesh, node_name=f'mesh_{i}')

    else:  # .obj
        # Create a scene for multiple meshes
        transformed_scene = trimesh.Scene()
        for mesh in transformed_meshes:
            transformed_scene.add_geometry(mesh)

    # Create a pyrender scene
    scene = pyrender.Scene()

    # Convert and add all meshes to pyrender scene
    for mesh in transformed_meshes:
        if isinstance(mesh, trimesh.Trimesh):
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
            scene.add(mesh_pyrender)

    # Create a camera and add it to the scene
    camera = pyrender.PerspectiveCamera(yfov=FOV*1.2)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    # Add light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    # Create an off-screen renderer
    renderer = pyrender.OffscreenRenderer(width, height)

    # Split frames among rotations
    frames_per_rotation = n_frames // len(rotation_order)
    frames = []
    
    # Current accumulated rotation matrix
    current_rotation = np.eye(4)
    
    # Create frames for each rotation axis
    for axis in rotation_order:
        for i in range(frames_per_rotation):
            # Calculate rotation angle for this step
            angle = (i / frames_per_rotation) * 2 * np.pi
            # Get rotation matrix for current axis
            rotation = create_rotation_matrix(angle, axis)
            # Apply rotation to all mesh nodes
            for node in scene.get_nodes():
                if node.mesh is not None:
                    scene.set_pose(node, rotation @ current_rotation)
            # Render frame
            color, depth = renderer.render(scene)
            frames.append(Image.fromarray(color))
        
        # Update accumulated rotation for next axis
        current_rotation = create_rotation_matrix(2 * np.pi, axis) @ current_rotation

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000/30,  # 30 fps
        loop=0
    )

    # Clean up
    renderer.delete()
    print(f"GIF saved as {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate 3D model rotation animations')
    parser.add_argument('input_path', help='Input file or directory path')
    parser.add_argument('output_path', help='Output file or directory path for the GIF(s)')
    parser.add_argument('model_name', choices=['G3D', 'IM', 'LGM', 'TG', 'TSR'], help='Name of the 3D model')
    parser.add_argument('--type', choices=['glb', 'obj'], required=True, help='Type of 3D model files')
    parser.add_argument('--n-frames', type=int, default=60, help='Number of frames in the animation')
    parser.add_argument('--width', type=int, default=400, help='Width of the output animation')
    parser.add_argument('--height', type=int, default=400, help='Height of the output animation')
    parser.add_argument('--rotation-order', type=str, default='xyz',
                      help='Order of rotations (e.g., "xyz", "zyx", "xzy")')

    args = parser.parse_args()
    
    # Validate rotation order
    valid_axes = set('xyz')
    if not set(args.rotation_order).issubset(valid_axes) or len(args.rotation_order) > 3:
        parser.error("Rotation order must only contain x, y, and z")

    if os.path.isfile(args.input_path):
        # Create output directory for single file
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        create_animation(args.input_path, args.output_path, args.model_name, args.type, args.n_frames, 
                        args.width, args.height, args.rotation_order)
    else:
        # Create output directory for multiple files
        os.makedirs(args.output_path, exist_ok=True)
        for filename in os.listdir(args.input_path):
            if filename.endswith(f'.{args.type}'):
                input_file = os.path.join(args.input_path, filename)
                output_file = os.path.join(args.output_path, f'{os.path.splitext(filename)[0]}.gif')
                create_animation(input_file, output_file, args.model_name, args.type, args.n_frames,
                               args.width, args.height, args.rotation_order)

if __name__ == '__main__':
    main()


# python utils/gen_animation.py out/Gen3Diffusion/color_mesh/IMG_SEG_01.glb output_gen3d.gif G3D --type glb --rotation-order yx --n-frames 270
# python utils/gen_animation.py out/instant-mesh-large/color_mesh/IMG_SEG_01.obj output_im.gif IM --type obj --rotation-order yx --n-frames 270
# python utils/gen_animation.py out/LGM/color_mesh/IMG_SEG_01.glb IMG_SEG_01.gif LGM --type glb --rotation-order yx --n-frames 270

# python utils/gen_animation.py out/TriplaneGaussian/color_mesh/IMG_SEG_01.glb output_tg.gif TG --type glb --rotation-order yx --n-frames 270
# python utils/gen_animation.py out/TripoSR/color_mesh/IMG_SEG_01.obj IMG_SEG_01.gif TSR --type obj --rotation-order yx --n-frames 270




# python utils/gen_animation.py out/Gen3Diffusion/color_mesh/ final_gifs/Gen3Diffusion G3D --type glb --rotation-order yx --n-frames 270
# python utils/gen_animation.py out/instant-mesh-large/color_mesh/ final_gifs/IM IM --type obj --rotation-order yx --n-frames 270
# python utils/gen_animation.py out/LGM/color_mesh/  final_gifs/LGM LGM --type glb --rotation-order yx --n-frames 270
# python utils/gen_animation.py out/TriplaneGaussian/color_mesh/ final_gifs/TriplaneGaussian TG --type glb --rotation-order yx --n-frames 270
# python utils/gen_animation.py out/TripoSR/color_mesh/ final_gifs/TripoSR TSR --type obj --rotation-order yx --n-frames 270