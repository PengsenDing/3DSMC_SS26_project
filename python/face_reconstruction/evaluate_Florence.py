# Adapted from GANFit, see: https://github.com/barisgecer/GANFit/tree/master
import os
import glob
import numpy as np
import json
import open3d as o3d
from collections import defaultdict
import trimesh
import pyrender
import matplotlib.pyplot as plt
from PIL import Image
import copy
import re
from scipy.spatial.transform import Rotation as R

# Landmark ids in MICC Florence
LEFT_EYE_IDX = 36
RIGHT_EYE_IDX = 45
NOSE_IDX = 30
# Vertex ids in BFM 2019
NOSE_IDX_BFM = 15841

def parse_pose(folder_name):
    match = re.search(r"yaw_(-?\d+)_pitch_(-?\d+)", folder_name)
    if match:
        yaw = int(match.group(1))
        pitch = int(match.group(2))
        return yaw, pitch
    raise ValueError(f"Cannot parse pose: {folder_name}")

class SyntheticBenchmark:
    def __init__(self, registration_path):
        self.registration_path = registration_path

    def _load_as_o3d(self, path):
        """Robustly load a mesh using trimesh and convert to Open3D."""
        # Force='mesh' ensures we get a mesh even if the file format is weird
        mesh = trimesh.load(path, force='mesh')
        
        # Convert to Open3D TriangleMesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        
        # Compute normals (required for Point-to-Plane ICP)
        o3d_mesh.compute_vertex_normals()
        return o3d_mesh
    
    import json

    def _get_landmark_pos_from_ljson(self, landmark_idx, subject_id, load_frontal_2=False):
        """
        Reads the .ljson file to find the nose tip coordinate.
        You need to adjust 'MICC_landmarks_path' to where your LJSON files actually are.
        """
        subject = f"subject_{subject_id:02d}"
        # Adjust this path pattern to match your folder structure
        ljson_path = os.path.join("datasets/Florence/MICC_landmarks", subject, "Model", "frontal1", "obj")
        if load_frontal_2:
            # Fallback to frontal2 if frontal1 is not available
            ljson_path = ljson_path.replace("frontal1", "frontal2")
        
        # Search for .ljson file in the directory
        search_pattern = os.path.join(ljson_path, "*.ljson")
        files = glob.glob(search_pattern)
        
        with open(files[0], 'r') as f:
            data = json.load(f)
        
        points = np.array(data['groups']['LJSON']['landmarks']['points'])
        
        return points[landmark_idx]
    

    def get_iod(mesh_vertices):
        # Calculate Euclidean distance between the two eye corner vertices
        p1 = mesh_vertices[LEFT_EYE_IDX]
        p2 = mesh_vertices[RIGHT_EYE_IDX]
        return np.linalg.norm(p1 - p2)
    
    def visualize_landmarks(self, reg_mesh, landmark_indices, save_dir, subject_id):
        """
        Collects all landmark points and saves them as a single PLY file.
        """
        landmark_points = []
        
        # Collect all points into a list
        for landmark_idx in landmark_indices:
            pos = self._get_landmark_pos_from_ljson(landmark_idx, subject_id)
            landmark_points.append(pos)
        
        # Convert to NumPy array
        pts_array = np.array(landmark_points)

        # Create the point cloud
        landmark_pcd = o3d.geometry.PointCloud()
        landmark_pcd.points = o3d.utility.Vector3dVector(pts_array)
        
        # Paint red for visibility
        landmark_pcd.paint_uniform_color([1, 0, 0]) 
        
        # Save
        output_path = os.path.join(save_dir, f"subject_{subject_id:02d}_landmarks.ply")
        o3d.io.write_point_cloud(output_path, landmark_pcd)
        
        print(f"Saved {len(landmark_points)} landmarks to {output_path}")


    def save_heatmap_with_colorbar(image, save_path, max_error, cmap_name="jet"):
        """
        Save a rendered heatmap image with a colorbar legend.

        Args:
            image: numpy RGB image from pyrender (H, W, 3)
            save_path: output path
            max_error: maximum error represented by the colormap (same as heatmap normalization)
            cmap_name: matplotlib colormap name
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from PIL import Image
        import numpy as np

        heat_img = Image.fromarray(image)

        # Create colorbar
        fig, ax = plt.subplots(figsize=(3, 5), dpi=100)

        norm = Normalize(vmin=0, vmax=max_error)
        sm = ScalarMappable(norm=norm, cmap=cmap_name)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Vertex error (mm)")

        ax.remove()

        # Convert matplotlib figure to numpy image
        fig.canvas.draw()

        width, height = fig.canvas.get_width_height()
        colorbar_img = np.frombuffer(
            fig.canvas.buffer_rgba(),
            dtype=np.uint8
        ).reshape(height, width, 4)

        plt.close(fig)

        colorbar_img = Image.fromarray(colorbar_img[:, :, :3])

        # Combine images
        combined = Image.new(
            "RGB",
            (
                heat_img.width + colorbar_img.width,
                max(heat_img.height, colorbar_img.height)
            ),
            (255, 255, 255)
        )

        combined.paste(heat_img, (0, 0))
        combined.paste(colorbar_img, (heat_img.width, 0))

        combined.save(save_path)

    def _render_debug_views(self, reg_mesh, rec_mesh_aligned, subject_id, save_dir, yaw, pitch):
        bg_color = [1.0, 1.0, 1.0]
        baseColorFactor=[0.60, 0.57, 0.55, 1.0] 
        # Convert Open3D meshes to Trimesh
        gt_tri = trimesh.Trimesh(vertices=np.asarray(reg_mesh.vertices), faces=np.asarray(reg_mesh.triangles))
        rec_tri = trimesh.Trimesh(vertices=np.asarray(rec_mesh_aligned.vertices), faces=np.asarray(rec_mesh_aligned.triangles))

        # Get the closest points on the GT mesh for each reconstruction vertex
        _, distances, _ = trimesh.proximity.closest_point(gt_tri, rec_tri.vertices)

        def apply_base_transform(base_mesh, scale_factor=None):
            pre_rot = R.from_euler('x', [-45], degrees=True)
            base_mesh.apply_transform(np.pad(pre_rot.as_matrix(), ((0,1),(0,1)), mode='constant'))

            # Calculate the center of the mesh
            centroid = base_mesh.centroid
            # Translate the mesh to the origin (0, 0, 0)
            base_mesh.apply_translation(-centroid)
            # Scale the mesh so the largest dimension is 1.0
            # This ensures it fits perfectly in the standard 'pyrender' view
            
            if scale_factor is None:
                scale_factor = 1.0 / base_mesh.extents.max()
            base_mesh.apply_scale(scale_factor)
            return base_mesh, scale_factor
        # Apply the exact same base rotation to GT and reconstruction mesh to match image
        gt_tri, scale_factor = apply_base_transform(gt_tri)
        rec_tri, _ = apply_base_transform(rec_tri, scale_factor)

        rotation = R.from_euler(
            'yx',
            [yaw, pitch],
            degrees=True
        )

        transform = np.eye(4)
        transform[:3,:3] = rotation.as_matrix()

        gt_tri.apply_transform(transform)
        rec_tri.apply_transform(transform)
        
        # Compute Vertex Errors for Heatmap
        max_err = 6.0 
        norm_err = np.clip(distances / max_err, 0, 1)
        cmap = plt.get_cmap('jet')
        vertex_colors = (cmap(norm_err)[:, :3] * 255).astype(np.uint8)
        rec_tri.visual.vertex_colors = vertex_colors

        # Setup Pyrender Camera & Lighting
        r = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 1.0  # Camera placed 2 units away on Z-axis
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        
        # RENDER A: Heatmap View
        scene_heat = pyrender.Scene(bg_color=bg_color, ambient_light=[0.3, 0.3, 0.3])
        scene_heat.add(pyrender.Mesh.from_trimesh(rec_tri))
        scene_heat.add(cam, pose=cam_pose)
        scene_heat.add(light, pose=cam_pose)
        
        color_heat, _ = r.render(scene_heat)
        Image.fromarray(color_heat).save(os.path.join(save_dir, f"{subject_id}_heatmap.png"))

        SyntheticBenchmark.save_heatmap_with_colorbar(
            color_heat,
            os.path.join(save_dir, f"{subject_id}_heatmap_color_bar.png"),
            max_error=max_err,
            cmap_name="jet"
        )
        
        # RENDER B: GT mesh (Illumination only, no texture)
        gt_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=baseColorFactor, metallicFactor=0.1, roughnessFactor=0.9)
        rec_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=baseColorFactor, metallicFactor=0.1, roughnessFactor=0.9)
        
        scene_overlap = pyrender.Scene(bg_color=bg_color, ambient_light=[0.3, 0.3, 0.3])
        scene_overlap.add(pyrender.Mesh.from_trimesh(gt_tri, material=gt_material))
        scene_overlap.add(cam, pose=cam_pose)
        scene_overlap.add(light, pose=cam_pose)
        
        color_overlap, _ = r.render(scene_overlap)
        Image.fromarray(color_overlap).save(os.path.join(save_dir, f"{subject_id}_gt.png"))
        

        # RENDER C: Recon mesh Illumination only, no texture)
        scene_rec = pyrender.Scene(bg_color=bg_color, ambient_light=[0.3, 0.3, 0.3])
        scene_rec.add(pyrender.Mesh.from_trimesh(rec_tri, material=rec_material))
        scene_rec.add(cam, pose=cam_pose)
        scene_rec.add(light, pose=cam_pose)
        color_rec, _ = r.render(scene_rec)
        Image.fromarray(color_rec).save(os.path.join(save_dir, f"{subject_id}_rec.png"))
        r.delete()
        print(f"Saved debug renders to {save_dir}")

    def _apply_mask(self, pcd, nose_tip, tight_mask):
        """Helper to mask the point cloud based on distance to the nose."""
        threshold = 60 if tight_mask else 95        # Mask threshold in mm
        
        # Calculate distances of all points to the nose tip
        pts = np.asarray(pcd.points)
        distances = np.linalg.norm(pts - nose_tip, axis=1)
        
        # Mask points
        mask = distances < threshold
        return pcd.select_by_index(np.where(mask)[0])

    @staticmethod
    def robust_registration(rec_pcd, reg_pcd, nose_tip_rec, nose_tip_reg):
        """
        Performs multi-stage ICP registration.
        """
        rec_aligned = o3d.geometry.PointCloud(rec_pcd)
        
        # Pre-alignment shift
        shift = nose_tip_reg - nose_tip_rec
        rec_aligned.translate(shift)
        
        total_transform = np.eye(4)
        total_transform[:3, 3] = shift
        
        stages = [20.0, 5.0, 1.0, 0.5]
        for threshold in stages:
            result = o3d.pipelines.registration.registration_icp(
                rec_aligned, reg_pcd, 
                threshold, 
                np.identity(4), 
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            rec_aligned.transform(result.transformation)
            total_transform = result.transformation @ total_transform
            
        return rec_aligned, total_transform

    def bi_point2plane_ICP(self, reg_mesh, rec_mesh, tight_mask, subject_id, iod, debug=True, yaw=0, pitch=0, pose="pose"):
        """Calculates distance with diagnostic printing."""

        # Mask the point clouds based on nose tip proximity (rec, reg have different nose tip indices)
        nose_tip_reg = self._get_landmark_pos_from_ljson(NOSE_IDX, subject_id)
        rec_vertices = np.asarray(rec_mesh.vertices)
        nose_tip_rec = rec_vertices[NOSE_IDX_BFM]

        # Convert Meshes to Point Clouds
        reg_pcd = reg_mesh.sample_points_uniformly(number_of_points=10000, use_triangle_normal=True)
        rec_pcd = rec_mesh.sample_points_uniformly(number_of_points=10000, use_triangle_normal=True)
        
        # Apply Mask
        reg_pcd = self._apply_mask(reg_pcd, nose_tip_reg, tight_mask)
        rec_pcd = self._apply_mask(rec_pcd, nose_tip_rec, tight_mask)

        # Perform robust ICP using point-to-plane metric
        rec_trans_pcd, icp_transform = SyntheticBenchmark.robust_registration(rec_pcd, reg_pcd, nose_tip_rec, nose_tip_reg)
        
        # Calculate Mean Distance (RMSE)
        distances = rec_trans_pcd.compute_point_cloud_distance(reg_pcd)
        mean_dist_1 = np.mean(distances)
        rmse_1 = np.sqrt(np.mean(np.asarray(mean_dist_1)**2))

        reg_trans_pcd, _ = SyntheticBenchmark.robust_registration(reg_pcd, rec_pcd, nose_tip_reg, nose_tip_rec)

        # Calculate Mean Distance (RMSE)
        distances = reg_trans_pcd.compute_point_cloud_distance(rec_pcd)
        mean_dist_2 = np.mean(distances)
        rmse_2 = np.sqrt(np.mean(np.asarray(mean_dist_2)**2))

        rmse = (rmse_1 + rmse_2) / 2.0
        nme = rmse / iod
        
        if debug:
            save_dir = os.path.join("outputs", "debug_diagnostics", f"subject_{subject_id:02d}", pose)
            os.makedirs(save_dir, exist_ok=True)
            # Create a fully transformed copy of the reconstruction mesh for rendering
            rec_mesh_aligned = copy.deepcopy(rec_mesh)
            rec_mesh_aligned.transform(icp_transform)

            # Render the heatmap and overlapping view
            self._render_debug_views(reg_mesh, rec_mesh_aligned, subject_id, save_dir, yaw, pitch)

            # Visualize landmarks for Inter Ocular Distance
            self.visualize_landmarks(reg_pcd, [36, 45], save_dir, subject_id)
            # Save the registered Ground Truth
            o3d.io.write_point_cloud(os.path.join(save_dir, f"{subject_id}_reg.ply"), rec_pcd)
            # Save the resgistered nose tip for debugging
            nose_tip_pcd = o3d.geometry.PointCloud()
            nose_tip_pcd.points = o3d.utility.Vector3dVector([nose_tip_reg])
            o3d.io.write_point_cloud(os.path.join(save_dir, f"{subject_id}_nose_tip.ply"), nose_tip_pcd)
            # Save the aligned Reconstruction
            o3d.io.write_point_cloud(os.path.join(save_dir, f"{subject_id}_rec.ply"), reg_pcd)
            
        
        return rmse, nme

    def _load_gt_mesh(self, subject_id):
        """Load GT mesh using glob to handle wildcard pattern."""
        subject = f"subject_{subject_id:02d}"
        
        # Define the pattern
        search_pattern = os.path.join(self.registration_path, subject, "Model", "frontal1", "obj", "*.obj")
        files = glob.glob(search_pattern)
        
        # Handle if error while loading
        mesh = None
        try:
            if files:
                mesh = self._load_as_o3d(files[0])
        except Exception as e:
            # Fallback to frontal2 if frontal1 fails
            print(f"Warning: Failed to load frontal1 for {subject}. Trying frontal2.")
            search_pattern = os.path.join(self.registration_path, subject, "Model", "frontal2", "obj", "*.obj")
            files = glob.glob(search_pattern)
            mesh = None
            try:
                if files:
                    mesh = self._load_as_o3d(files[0])
            except Exception as e2:
                print(f"Error: Failed to load frontal2 for {subject}. Exception: {e2}")
                return None
            
        if mesh is None:
            print(f"Error: No valid mesh found for {subject}.")
        return mesh

    def run_evaluation(self, reconstruction_root, debug=False):
        """Iterates through subjects, averages poses per subject, then averages subjects."""
        # Use a dictionary to group errors by subject
        subject_rmse_map = defaultdict(list)
        subject_nme_map = defaultdict(list)
        failed_runs = []
        
        subjects = sorted([d for d in os.listdir(reconstruction_root) if d.startswith("subject_")])
        
        for subject in subjects[-6:]:
            print(f"Evaluating subject {subject}")
            subj_id = int(subject.split("_")[1])
            subj_dir = os.path.join(reconstruction_root, subject)
            poses = sorted([d for d in os.listdir(subj_dir) if d.startswith("pose_")])
            
            reg = self._load_gt_mesh(subj_id)

            left_eye = self._get_landmark_pos_from_ljson(LEFT_EYE_IDX, subj_id)
            right_eye = self._get_landmark_pos_from_ljson(RIGHT_EYE_IDX, subj_id)
            iod = np.linalg.norm(left_eye - right_eye)
            
            for pose in poses:
                rec_path = os.path.join(subj_dir, pose, "face.ply")
                yaw, pitch = parse_pose(pose)
                if not os.path.exists(rec_path):
                    failed_runs.append(f"{subject}/{pose}")
                    continue

                try:
                    rec = self._load_as_o3d(rec_path)
                    rmse, nme = self.bi_point2plane_ICP(reg, rec, False, subj_id, iod, debug=debug, yaw=yaw, pitch=pitch, pose=pose)
                    subject_rmse_map[subject].append(rmse)
                    subject_nme_map[subject].append(nme)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    failed_runs.append(f"{subject}/{pose}")

        # Now, aggregate by subject
        subject_rmse_means = []
        subject_nme_means = []
        for subj_name, errors in subject_rmse_map.items():
            if len(errors) > 0:
                subject_rmse_means.append(np.mean(errors))
                print(f"Subject {subj_name} with RMSE: {subject_rmse_means[-1]}")

        for subj_name, errors in subject_nme_map.items():
            if len(errors) > 0:
                subject_nme_means.append(np.mean(errors))
                print(f"Subject {subj_name} with NME: {subject_nme_means[-1]}")
        
        # Calculate the mean and std of the subject means
        overall_rmse_mean = np.mean(subject_rmse_means)
        overall_rmse_std = np.std(subject_rmse_means)
        overall_nme_mean = np.mean(subject_nme_means)
        overall_nme_std = np.std(subject_nme_means)
        
        return overall_rmse_mean, overall_rmse_std, overall_nme_mean, overall_nme_std, failed_runs

if __name__ == "__main__":
    REG_PATH = "datasets/Florence/Original"
    REC_PATH = "outputs/reconstructions/Florence"
    
    benchmark = SyntheticBenchmark(REG_PATH)
    mean_rmse, std_rmse, mean_nme, std_nme, failed_runs = benchmark.run_evaluation(REC_PATH, debug=True)
    
    print(f"\n--- Final Results (Trimesh + Open3D) ---")
    print(f"Global Mean RMSE: {mean_rmse:.3f} ± {std_rmse:.3f} mm")
    print(f"Global Mean NME: {mean_nme * 100:.3f}% ± {std_nme * 100:.3f}%")
    print(f"Failed Runs: {len(failed_runs)}")
    for run in failed_runs:
        print(f"  - {run}")