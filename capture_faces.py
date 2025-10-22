import cv2
import os
import face_recognition
import numpy as np
import time
import sys
import signal
from PIL import Image

# ---------------- Settings ----------------
DATASET_PATH = "dataset"
IMAGE_COUNT = 50
CAMERA_INDEX = 0
WARMUP_FRAMES = 200
BAD_FRAME_REOPEN_THRESHOLD = 80
BAD_FRAME_PRINT_STEP = 10
WARMUP_SLEEP = 0.02
# ------------------------------------------

# --- get name ---
PERSON_NAME = ""
while not PERSON_NAME:
    PERSON_NAME = input("Enter your name: ").strip()
    if not PERSON_NAME:
        print("Name cannot be empty.")

person_path = os.path.join(DATASET_PATH, PERSON_NAME)
os.makedirs(person_path, exist_ok=True)
print(f"Directory: {person_path}")

# load Haar cascade (OpenCV fallback)
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def open_camera(index):
    print(f"Opening camera at index {index} with DSHOW backend...")
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print("DSHOW failed — trying default backend...")
        cap = cv2.VideoCapture(index)
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception:
        pass
    return cap

def warmup_camera(cap, tries=WARMUP_FRAMES):
    print("Warming up camera...")
    for i in range(tries):
        ret, frame = cap.read()
        if not ret or frame is None or getattr(frame, "size", 0) == 0:
            if i % BAD_FRAME_PRINT_STEP == 0:
                print(f"Warmup: bad frame {i+1}/{tries}")
            time.sleep(WARMUP_SLEEP)
            continue
        try:
            _ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print("Camera warmup successful. RGB frame acquired.")
            return True
        except (cv2.error, RuntimeError):
            if i % BAD_FRAME_PRINT_STEP == 0:
                print(f"Warmup: unsupported frame type {i+1}/{tries}")
            time.sleep(WARMUP_SLEEP)
            continue
    return False

cap = open_camera(CAMERA_INDEX)
if not cap.isOpened():
    print("Could not open webcam. Check index or other apps using camera.")
    sys.exit(1)

if not warmup_camera(cap):
    print(f"Failed to warm up camera after {WARMUP_FRAMES} attempts.")
    cap.release()
    sys.exit(1)

print("\nStarting video capture...")
print("Press 's' to save (only saved when exactly 1 face is found). Press 'q' to quit.")
print(f"Need {IMAGE_COUNT} images.")

saved_count = 0
consecutive_bad_frames = 0

def handle_sigint(signum, frame):
    print("\nInterrupted. Exiting...")
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

try:
    while saved_count < IMAGE_COUNT:
        ret, frame = cap.read()

        if not ret or frame is None or getattr(frame, "size", 0) == 0:
            consecutive_bad_frames += 1
            if consecutive_bad_frames % BAD_FRAME_PRINT_STEP == 0:
                print(f"Warning: empty/failed frame. Consecutive bad: {consecutive_bad_frames}")
            if consecutive_bad_frames >= BAD_FRAME_REOPEN_THRESHOLD:
                print("Too many bad frames. Reopening camera...")
                cap.release()
                time.sleep(0.5)
                cap = open_camera(CAMERA_INDEX)
                if not cap.isOpened() or not warmup_camera(cap):
                    print("Failed to recover camera. Exiting.")
                    break
                consecutive_bad_frames = 0
            time.sleep(0.01)
            continue

        # normalize: handle 4-channel, non-uint8, single-channel
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            # depth/IR single channel — skip
            print(f"Skipping single-channel frame (shape={frame.shape}, dtype={frame.dtype})")
            consecutive_bad_frames += 1
            time.sleep(0.01)
            continue

        if len(frame.shape) == 3 and frame.shape[2] == 4:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            except cv2.error:
                frame = frame[:, :, :3]

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                frame = np.clip(frame, 0.0, 1.0)
                frame = (frame * 255.0).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # convert to RGB
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error:
            print("Failed BGR->RGB conversion, skipping this frame.")
            consecutive_bad_frames += 1
            continue

        # MAKE contiguous + explicit full copy (this helps some dlib build issues)
        rgb_frame = np.ascontiguousarray(rgb_frame).copy()

        # reset bad counter since we have a usable rgb_frame
        consecutive_bad_frames = 0

        # --- Try face_recognition first (with a PIL roundtrip fallback) ---
        face_locations = None
        detector_used = None

        # Attempt 1: direct call
        try:
            face_locations = face_recognition.face_locations(rgb_frame)
            detector_used = "face_recognition"
        except Exception as e1:
            # Attempt 2: PIL roundtrip (sometimes resolves hidden layout issues)
            try:
                pil = Image.fromarray(rgb_frame)
                arr = np.array(pil)
                arr = np.ascontiguousarray(arr).copy()
                face_locations = face_recognition.face_locations(arr)
                detector_used = "face_recognition (PIL roundtrip)"
            except Exception as e2:
                # Attempt 3: Haar cascade fallback (use grayscale)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    # Convert OpenCV rectangles to (top, right, bottom, left) style
                    face_locations = []
                    for (x, y, w, h) in faces:
                        top, left, bottom, right = y, x, y + h, x + w
                        face_locations.append((top, right, bottom, left))
                    detector_used = "opencv_haar (fallback)"
                else:
                    detector_used = "none"

        # draw results
        if face_locations is None:
            status_text = "Error: Detector failed"
            status_color = (0, 0, 255)
        elif len(face_locations) == 1:
            (top, right, bottom, left) = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            status_text = f"Face Detected: OK ({detector_used})"
            status_color = (0, 255, 0)
        elif len(face_locations) > 1:
            status_text = f"Error: Multiple faces ({detector_used})"
            status_color = (0, 0, 255)
        else:
            status_text = f"Error: No face ({detector_used})"
            status_color = (0, 0, 255)

        progress_text = f"Saved: {saved_count}/{IMAGE_COUNT}"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow("Capture Faces - Press 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if face_locations is not None and len(face_locations) == 1:
                image_path = os.path.join(person_path, f"{saved_count + 1:03d}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Saved: {image_path}  (detector: {detector_used})")
                saved_count += 1
            else:
                print(f"Not saved: {status_text}")
        elif key == ord('q'):
            print("Quitting.")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting...")

finally:
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print(f"Captured {saved_count} images for {PERSON_NAME}.")
