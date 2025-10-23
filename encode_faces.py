import os
import pickle
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image, ImageFile
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm

# Allow loading truncated images (useful for some datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------- Settings --------
DATASET_PATH = "dataset"
OUTPUT_FILE = "encodings_facenet.pickle"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
BATCH_SIZE = 32
BATCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_FACE_SIZE = 20
RESIZE_DIM = (600, 600) # Target size for resizing (Width, Height)
DELETE_FAILED_IMAGES = True
# --------------------

print(f"Using device: {BATCH_DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Resizing images to: {RESIZE_DIM}")
if DELETE_FAILED_IMAGES:
    print("\nWARNING: DELETE_FAILED_IMAGES is set to True.")
    print("Images causing errors (load, resize, validation, MTCNN fail) OR where no face is detected will be DELETED.")
else:
    print("\nDELETE_FAILED_IMAGES is False. Failed/undetected images will be skipped but not deleted.")
print("-" * 30)


# --- Helper Function for Batch Processing ---
def process_batch(
    image_paths: List[str],
    person_names: List[str],
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: str,
    resize_dim: Tuple[int, int],
    delete_failed: bool # Pass the flag
) -> Tuple[List[List[float]], List[str]]:
    """Loads images, resizes, detects faces (batch with single fallback), calculates embeddings."""
    batch_encodings = []
    batch_names_processed = []
    images_pil_ready = []
    valid_indices = [] # Indices of images successfully loaded, resized, AND validated

    # 1. Load, Resize, and Validate images
    for i, path in enumerate(image_paths):
        failed_path = path # Keep track for potential deletion
        try:
            img = Image.open(path).convert("RGB")
            img_resized = img.resize(resize_dim, Image.Resampling.LANCZOS)
            img_array = np.array(img_resized)
            if img_array.ndim != 3 or img_array.shape[2] != 3:
                print(f"\n  - WARN: Invalid shape {img_array.shape} for {path} after RGB conversion. Skipping.")
                if delete_failed:
                    try:
                        os.remove(failed_path)
                        print(f"  - DELETED invalid shape image: {failed_path}")
                    except OSError as delete_e:
                        print(f"  - FAILED TO DELETE {failed_path}: {delete_e}")
                continue # Skip this image
            images_pil_ready.append(img_resized)
            valid_indices.append(i) # Mark index as valid only after passing checks
        except Exception as e:
            print(f"\n  - WARN: Could not open/resize/validate {path}: {e}. Skipping in this batch.")
            if delete_failed:
                try:
                    os.remove(failed_path)
                    print(f"  - DELETED failed image (load/resize error): {failed_path}")
                except OSError as delete_e:
                    print(f"  - FAILED TO DELETE {failed_path}: {delete_e}")

    if not images_pil_ready:
        return [], []

    face_tensors_list = [None] * len(images_pil_ready) # Initialize list for results

    # 2. Try Batch Detection
    batch_failed = False
    try:
        detected_tensors = mtcnn(images_pil_ready)
        if detected_tensors is not None:
             if len(detected_tensors) == len(images_pil_ready):
                  face_tensors_list = detected_tensors
                  # print("  - Batch detection successful.") # Optional Debug
             else:
                  print("\n  - WARN: MTCNN batch output length mismatch. Falling back to single processing.")
                  batch_failed = True
        else:
            # This can happen if no faces are found in *any* image of the batch
            print("\n  - INFO: MTCNN batch returned None (no faces detected in batch or potential error). Will check individually.")
            # Set batch_failed to True to force individual checks, including potential deletion
            batch_failed = True # Treat 'None' for whole batch same as error for fallback/deletion logic

    except Exception as e:
        error_str = str(e)
        if "setting an array element with a sequence" in error_str or "inhomogeneous" in error_str:
            print(f"\n  - WARN: MTCNN batch failed ({type(e).__name__}). Falling back to single image processing for this batch.")
            batch_failed = True
        else:
            print(f"\n  - WARN: Unexpected MTCNN runtime error on batch: {e}. Skipping batch.")
            # Cannot easily delete here as error affects the whole batch run
            return [], []

    # --- Fallback to Single Image Processing ---
    if batch_failed:
        # If batch processing itself failed OR if it returned None for the whole batch
        print("  - Processing batch images individually...")
        temp_tensors = []
        for img_idx, pil_img in enumerate(images_pil_ready):
            original_index = valid_indices[img_idx]
            failed_path = image_paths[original_index] # Get the specific path
            try:
                single_face_tensor = mtcnn(pil_img) # Process one image
                temp_tensors.append(single_face_tensor)
            except Exception as single_e:
                print(f"\n  - WARN: MTCNN error on single image {failed_path}: {single_e}. Skipping image.")
                if delete_failed:
                    try:
                        os.remove(failed_path)
                        print(f"  - DELETED failed image (MTCNN single error): {failed_path}")
                    except OSError as delete_e:
                        print(f"  - FAILED TO DELETE {failed_path}: {delete_e}")
                temp_tensors.append(None) # Add None on error
        face_tensors_list = temp_tensors
    # --- End Fallback ---


    # 3. Filter out Nones, prepare valid tensors, and handle deletion for no-face-detected cases
    valid_face_tensors = []
    valid_original_indices = []

    for i, tensor in enumerate(face_tensors_list):
        original_index = valid_indices[i] # Map back to original batch index
        if tensor is not None and isinstance(tensor, torch.Tensor):
            valid_face_tensors.append(tensor)
            valid_original_indices.append(original_index)
        else:
            # Handle deletion for images where no face was detected (tensor is None)
            failed_path = image_paths[original_index]
            print(f"\n  - INFO: No face detected or tensor invalid in {failed_path}. Skipping encoding.")
            if delete_failed:
                try:
                    os.remove(failed_path)
                    print(f"  - DELETED image (no face detected/invalid tensor): {failed_path}")
                except OSError as delete_e:
                    print(f"  - FAILED TO DELETE {failed_path}: {delete_e}")

    if not valid_face_tensors:
        return [], []

    # 4. Stack face tensors into a batch and move to device
    try:
        batch_face_tensor = torch.stack(valid_face_tensors).to(device)
    except Exception as e:
        print(f"\n  - WARN: Error stacking face tensors: {e}. Skipping results from this batch.")
        # Cannot delete here as error is for the batch stack
        return [], []

    # 5. Calculate embeddings for the batch
    embeddings_list = []
    batch_names_processed = []
    try:
        with torch.no_grad():
            embeddings_batch = resnet(batch_face_tensor)
        embeddings_list = embeddings_batch.cpu().numpy().tolist()
        batch_names_processed = [person_names[i] for i in valid_original_indices]
        batch_encodings = embeddings_list

    except Exception as e:
        print(f"\n  - WARN: Embedding error on batch: {e}. Skipping results from this batch.")
        # Cannot delete here as error applies to the whole batch embedding step
        return [], []

    return batch_encodings, batch_names_processed

# --- Main Script ---
start_total_time = time.time()

# Create detector & embedding model
print("Initializing models...")
try:
    mtcnn = MTCNN(
        keep_all=False,
        min_face_size=MIN_FACE_SIZE,
        device=BATCH_DEVICE,
        select_largest=False # Keep this False to ensure single face as expected
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(BATCH_DEVICE)
except Exception as e:
     raise SystemExit(f"Error initializing models: {e}")
print("Models initialized.")


all_encodings: List[List[float]] = []
all_names: List[str] = []
image_paths_to_process: List[str] = []
names_to_process: List[str] = []

if not os.path.exists(DATASET_PATH):
    raise SystemExit(f"Dataset path '{DATASET_PATH}' does not exist.")

print("Collecting image paths...")
# Collect all image paths first
initial_image_count = 0
for person in sorted(os.listdir(DATASET_PATH)):
    person_dir = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_dir):
        continue

    for fname in sorted(os.listdir(person_dir)):
        if os.path.splitext(fname.lower())[1] in IMAGE_EXTS:
            path = os.path.join(person_dir, fname)
            image_paths_to_process.append(path)
            names_to_process.append(person)
            initial_image_count += 1

print(f"Found {initial_image_count} images to process.") # Use initial count here

print("Processing images in batches...")
processed_image_count = 0 # Track total attempts across batches
# Process in batches using tqdm for progress
for i in tqdm(range(0, len(image_paths_to_process), BATCH_SIZE), desc="Encoding Batches"):
    batch_paths = image_paths_to_process[i:i + BATCH_SIZE]
    batch_names = names_to_process[i:i + BATCH_SIZE]
    processed_image_count += len(batch_paths) # Count attempts

    # Process the current batch, passing the resize and delete flags
    batch_enc, batch_proc_names = process_batch(
        batch_paths, batch_names, mtcnn, resnet, BATCH_DEVICE, RESIZE_DIM, DELETE_FAILED_IMAGES # Pass flag
    )

    all_encodings.extend(batch_enc)
    all_names.extend(batch_proc_names)


# Save results
end_total_time = time.time()
total_duration = end_total_time - start_total_time

print(f"\nEncoding complete in {total_duration:.2f} seconds.")
print(f"Attempted to process {initial_image_count} images found initially.") # Report based on initial scan
print(f"Successfully encoded {len(all_encodings)} faces.")

# Handle deletion summary
if DELETE_FAILED_IMAGES:
     remaining_image_count = 0
     for person in sorted(os.listdir(DATASET_PATH)):
         person_dir = os.path.join(DATASET_PATH, person)
         if not os.path.isdir(person_dir): continue
         for fname in os.listdir(person_dir):
             if os.path.splitext(fname.lower())[1] in IMAGE_EXTS:
                 remaining_image_count += 1
     deleted_count = initial_image_count - remaining_image_count
     print(f"Approximately {deleted_count} images were deleted due to errors or no face detection.")


if not all_encodings:
     print("\nNo faces were encoded. Check dataset and logs.")
else:
     print(f"\nSaving {len(all_encodings)} embeddings to {OUTPUT_FILE} ...")
     data: Dict[str, Any] = {"encodings": all_encodings, "names": all_names}
     try:
         with open(OUTPUT_FILE, "wb") as f:
             pickle.dump(data, f)
         print("Saved.")
     except Exception as e:
         print(f"Error saving pickle file: {e}")
