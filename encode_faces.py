import os
import pickle
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# -------- Settings --------
DATASET_PATH = "dataset"
OUTPUT_FILE = "encodings_facenet.pickle"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
BATCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------

print("Device:", BATCH_DEVICE)

# Create detector & embedding model
mtcnn = MTCNN(keep_all=False, device=BATCH_DEVICE)   # single face per image expected
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(BATCH_DEVICE)

encodings = []
names = []

if not os.path.exists(DATASET_PATH):
    raise SystemExit(f"Dataset path '{DATASET_PATH}' does not exist.")

for person in sorted(os.listdir(DATASET_PATH)):
    person_dir = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"Processing {person}...")
    for fname in sorted(os.listdir(person_dir)):
        if os.path.splitext(fname.lower())[1] not in IMAGE_EXTS:
            continue
        path = os.path.join(person_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  - Could not open {path}: {e}")
            continue

        # Detect & crop face. mtcnn returns a torch tensor (3,160,160) on device or None
        try:
            face_tensor = mtcnn(img)  # returns Tensor or None
        except Exception as e:
            print(f"  - MTCNN runtime error on {path}: {e}")
            face_tensor = None

        if face_tensor is None:
            print(f"  - No face detected in {path}. Skipping.")
            continue

        # face_tensor might be on CPU or device already depending on mtcnn; ensure correct dtype/device
        try:
            # If mtcnn returns a PIL-backed float tensor on CPU, move to proper device
            if isinstance(face_tensor, torch.Tensor):
                face_tensor = face_tensor.to(BATCH_DEVICE)
            else:
                # Unexpected type (rare), attempt conversion
                face_tensor = torch.tensor(np.array(face_tensor)).permute(2,0,1).unsqueeze(0).float().to(BATCH_DEVICE)
        except Exception as e:
            print(f"  - Error preparing tensor for {path}: {e}. Skipping.")
            continue

        # Compute embedding
        try:
            with torch.no_grad():
                if face_tensor.dim() == 3:
                    # add batch dim
                    input_tensor = face_tensor.unsqueeze(0)
                else:
                    input_tensor = face_tensor
                emb = resnet(input_tensor)  # (1,512)
                emb = emb.squeeze(0).cpu().numpy()  # to cpu numpy
        except Exception as e:
            print(f"  - Embedding error on {path}: {e}")
            continue

        encodings.append(emb.tolist())  # convert to list for pickle portability
        names.append(person)
        print(f"  - Encoded {fname}")

# Save results
print(f"\nEncoding complete. Saving {len(encodings)} embeddings to {OUTPUT_FILE} ...")
data = {"encodings": encodings, "names": names}
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)
print("Saved.")
