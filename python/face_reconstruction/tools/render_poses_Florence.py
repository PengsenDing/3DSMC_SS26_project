import os
import glob
import trimesh
import pyrender
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image

def render_pose(mesh, yaw, pitch):
    # Reset mesh orientation
    # Apply yaw (Y-axis) then pitch (X-axis)
    rot = R.from_euler('yx', [yaw, pitch], degrees=True)
    mesh.apply_transform(np.pad(rot.as_matrix(), ((0,1),(0,1)), mode='constant'))
    
    # Setup Scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    # Add mesh
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)
    
    # Add camera and light
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    # Place camera 2 units away on Z-axis
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 1.0 
    scene.add(camera, pose=cam_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)
    
    # Render
    r = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
    color, _ = r.render(scene)
    
    # Cleanup to avoid memory leaks
    r.delete()
    return color

# Main Configuration
input_dir = "datasets/Florence/Original"
output_dir = "datasets/Florence/rendered_gt_poses"

# Generate output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Angles (Adjusted for matching papers)
yaws = [-80, -40, 0, 40, 80]
pitches = [-15, 20, 25, 0]

# Find all subjects
subject_paths = sorted(glob.glob(os.path.join(input_dir, "subject_*")))

for subj_path in subject_paths:           # Process only the first subject for testing
    subj_name = os.path.basename(subj_path)
    # Look for the frontal1 scan
    mesh_files = glob.glob(os.path.join(subj_path, "Model/frontal1/obj/*.obj"))
    mesh_files_backup = glob.glob(os.path.join(subj_path, "Model/frontal2/obj/*.obj"))
    
    if not mesh_files:
        print(f"No frontal1 mesh found for {subj_name}, skipping.")
        continue
        
    mesh_file = mesh_files[0]
    # If frontal1 load fails, try frontal2
    try:
        base_mesh = trimesh.load(mesh_file)
    except Exception as e:
        print(f"Failed to load {mesh_file} for {subj_name}: {e}")
        if mesh_files_backup:
            try:
                base_mesh = trimesh.load(mesh_files_backup[0])
                print(f"Loaded backup frontal2 mesh for {subj_name}.")
            except Exception as e2:
                print(f"Failed to load backup frontal2 mesh for {subj_name}: {e2}")
                continue
        else:
            continue
    
    print(f"Processing {subj_name}...")

    # Small preprocessing
    # Rotate 45 degrees on X to fix the downward dip
    pre_rot = R.from_euler('x', [-45], degrees=True)
    base_mesh.apply_transform(np.pad(pre_rot.as_matrix(), ((0,1),(0,1)), mode='constant'))

    # Calculate the center of the mesh
    centroid = base_mesh.centroid
    # Translate the mesh to the origin (0, 0, 0)
    base_mesh.apply_translation(-centroid)
    # Scale the mesh so the largest dimension is 1.0
    # This ensures it fits perfectly in the standard 'pyrender' view
    scale_factor = 1.0 / base_mesh.extents.max()
    base_mesh.apply_scale(scale_factor)

    # Now your face is centered at (0,0,0) and fits inside a 1x1x1 box.
    
    for yaw in yaws:
        for pitch in pitches:
            # Create subfolder
            save_folder = os.path.join(output_dir, subj_name)
            os.makedirs(save_folder, exist_ok=True)
            
            # Create a fresh copy of the mesh for this specific rotation
            mesh_copy = base_mesh.copy()
            
            # Render
            img_array = render_pose(mesh_copy, yaw, pitch)
            
            # Save
            save_path = os.path.join(save_folder, f"pose_yaw_{yaw}_pitch_{pitch}.png")
            Image.fromarray(img_array).save(save_path)

print("Done generating synthetic poses.")