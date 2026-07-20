import os
import shutil
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Configuration
INPUT_ROOT = "datasets/Florence/rendered_gt_poses"
OUTPUT_ROOT = "outputs/reconstructions/Florence"
MAX_WORKERS = 5  # Number of parallel jobs

def process_single_image(img_path, subject_name, output_root):
    img_name = Path(img_path).stem
    # Unique temp name to prevent collisions
    temp_run_name = f"temp_{subject_name}_{img_name}"
    
    # Destination directory
    dest_dir = Path(output_root) / subject_name / img_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[{subject_name}] Starting {img_name}...")
    
    try:
        # Run reconstruction with a unique --name to avoid folder collision
        subprocess.run([
            "python", "apps/reconstruct.py", 
            "--diagnostics", 
            "--name", temp_run_name,
            img_path
        ], check=True)
        
        # Path to where the script dumped the files
        # It defaults to outputs/reconstructions/ + temp_run_name
        temp_output_dir = Path("outputs/reconstructions") / temp_run_name
        
        # Move files to the final destination
        if temp_output_dir.exists():
            for item in temp_output_dir.iterdir():
                shutil.move(str(item), str(dest_dir / item.name))
            shutil.rmtree(temp_output_dir) # Clean up
            print(f"[{subject_name}] Completed {img_name}.")
        else:
            print(f"[{subject_name}] Error: Expected output {temp_output_dir} not found.")

    except subprocess.CalledProcessError as e:
        print(f"[{subject_name}] Error processing {img_name}: {e}")
        # Delete temp output if it exists to avoid clutter
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)

def run_batch_processing(input_root, output_root):
    subjects = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])

    # Process subjects one by one, but images in parallel
    for subject in subjects:
        subject_path = os.path.join(input_root, subject)
        images = sorted(glob.glob(os.path.join(subject_path, "*.png")))
        
        print(f"\n--- Starting Subject: {subject} ({len(images)} images) ---")
        
        # Use ThreadPoolExecutor to run the reconstruction in parallel for each image
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map the process function across the list of images for this subject
            executor.map(lambda p: process_single_image(p, subject, output_root), images)

if __name__ == "__main__":
    run_batch_processing(INPUT_ROOT, OUTPUT_ROOT)