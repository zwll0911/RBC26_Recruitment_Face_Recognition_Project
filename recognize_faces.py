import face_recognition
import cv2
import os
import pickle
import imutils
import numpy as np
import time
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import subprocess
import sys
import signal

print("Starting video face recognition...")

# --- Settings ---
ENCODINGS_FILE = "encodings_facenet.pickle"
RESIZE_WIDTH = 500
TOLERANCE = 0.7
FRAME_SKIP = 5

# --- FFmpeg Settings ---
CAMERA_NAME = "USB2.0 HD UVC WebCam"  # <--- SET THIS TO YOUR CAMERA'S NAME
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 20

# --- General Settings ---
WARMUP_FRAMES = 200
BAD_FRAME_REOPEN_THRESHOLD = 80
BAD_FRAME_PRINT_STEP = 10
WARMUP_SLEEP = 0.02
# ------------------------------------------

# --- Load Encodings ---
print(f"Loading known face encodings from {ENCODINGS_FILE}...")
try:
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    # --- OPTIMIZATION: Convert known encodings to a single NumPy array for vectorized comparison ---
    known_encodings_np = np.array(data["encodings"])
    known_names = data["names"]
except FileNotFoundError:
    print(f"Error: Encodings file not found: {ENCODINGS_FILE}")
    print("Please run your 'encode_faces.py' first to create the file.")
    sys.exit(1)

# ------------------ FFmpeg pipe capture class ------------------
class FFMPEGPipeCapture:
    def __init__(self, dshow_name, width=640, height=480, fps=20):
        self.width = int(width)
        self.height = int(height)
        self.frame_size = self.width * self.height * 3 
        self._alive = False
        self.proc = None
        
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-f", "dshow",
            "-framerate", str(fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", f"video={dshow_name}",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-an", "-sn",
            "-f", "rawvideo", "-"
        ]
        
        popen_kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            
        try:
            self.proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found — ensure ffmpeg is installed and on your system's PATH.")
        except Exception as e:
            raise RuntimeError(f"Failed to start ffmpeg: {e!r}")

        time.sleep(0.5) 
        if self.proc.poll() is not None:
             stderr_output = self.proc.stderr.read().decode('utf-8', errors='ignore')
             raise RuntimeError(f"ffmpeg process failed on startup. Error: {stderr_output}")

        self._alive = True
        print(f"ffmpeg pipe started for 'video={dshow_name}'")

    def read(self, timeout=2.0):
        if not self.is_opened():
            return False, None
        
        buf = b""
        bytes_to_read = self.frame_size
        t0 = time.time()
        while len(buf) < bytes_to_read:
            try:
                chunk = self.proc.stdout.read(bytes_to_read - len(buf))
            except Exception:
                print("Error reading from ffmpeg stdout pipe.")
                self._alive = False
                return False, None

            if not chunk:
                print("ffmpeg process stdout pipe closed unexpectedly.")
                self._alive = False
                return False, None
                
            buf += chunk
            
            if timeout is not None and (time.time() - t0) > timeout:
                print(f"ffmpeg read timeout after {timeout}s")
                return False, None
                
        frame = np.frombuffer(buf, dtype=np.uint8)
        try:
            frame = frame.reshape((self.height, self.width, 3))
        except Exception as e:
            print(f"Error reshaping frame buffer: {e}")
            return False, None
            
        return True, frame

    def is_opened(self):
        if self._alive and self.proc and self.proc.poll() is None:
            return True
        self._alive = False
        return False

    def release(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    print("ffmpeg did not terminate, killing...")
                    self.proc.kill()
                    self.proc.wait(timeout=1.0)
            except Exception as e:
                print(f"Error during ffmpeg release: {e}")
            
            try:
                self.proc.stdout.close()
                self.proc.stderr.close()
            except Exception:
                pass

        self.proc = None
        self._alive = False

# ------------------ Camera Functions (Simplified) ------------------

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
            _ = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            print("Camera warmup successful. RGB frame acquired.")
            return True
        except (cv2.error, RuntimeError):
            if i % BAD_FRAME_PRINT_STEP == 0:
                print(f"Warmup: unsupported frame type {i+1}/{tries}")
            time.sleep(WARMUP_SLEEP)
            continue
    return False
# -----------------------------------------------------------------

# --- Load FaceNet Models ---
print("Loading FaceNet models...")
BATCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {BATCH_DEVICE}")
mtcnn = MTCNN(
    keep_all=True,
    device=BATCH_DEVICE,
    min_face_size=40
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(BATCH_DEVICE)
print("FaceNet models loaded.")

# --- Start Camera (Simplified) ---
try:
    print(f"Trying FFmpeg pipe for camera '{CAMERA_NAME}'...")
    cap = FFMPEGPipeCapture(CAMERA_NAME, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS)
    if not cap.is_opened():
        raise RuntimeError("FFMPEGPipeCapture .is_opened() returned False.")
    print("Successfully opened camera via FFmpeg pipe.")
except Exception as e:
    print(f"Failed to open camera via FFmpeg: {e!r}")
    print("Ensure ffmpeg is on PATH and CAMERA_NAME is correct.")
    sys.exit(1)


if not warmup_camera(cap):
    print(f"Failed to warm up camera after {WARMUP_FRAMES} attempts.")
    cap.release()
    sys.exit(1)

print("Webcam started. Starting recognition...")

# --- Graceful Exit Handler ---
def handle_sigint(signum, frame):
    print("\nInterrupted. Exiting...")
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# --- Optimization Variables ---
consecutive_bad_frames = 0
frame_count = 0
last_boxes = None
last_names = []
scale_y = 1.0
scale_x = 1.0

# --- Main Loop ---
try:
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None or frame.size == 0:
            consecutive_bad_frames += 1
            if consecutive_bad_frames % BAD_FRAME_PRINT_STEP == 0:
                print(f"Warning: empty/failed frame. Consecutive bad: {consecutive_bad_frames}")
            if consecutive_bad_frames >= BAD_FRAME_REOPEN_THRESHOLD:
                print("Too many bad frames. Reopening camera...")
                cap.release()
                time.sleep(0.5)
                try:
                    cap = FFMPEGPipeCapture(CAMERA_NAME, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS)
                except Exception:
                    cap = None
                            
                if not (cap and cap.is_opened() and warmup_camera(cap)):
                    print("Failed to recover camera. Exiting.")
                    break
                consecutive_bad_frames = 0
            time.sleep(0.01)
            continue
        
        display_frame = frame.copy()
        consecutive_bad_frames = 0 

        # --- --- OPTIMIZATION: FRAME SKIPPING --- ---
        if frame_count % FRAME_SKIP == 0:

            # --- OPTIMIZATION: Resize only when processing ---
            process_frame = imutils.resize(display_frame, width=RESIZE_WIDTH)
            
            scale_y = display_frame.shape[0] / process_frame.shape[0]
            scale_x = display_frame.shape[1] / process_frame.shape[1]

            boxes = None
            current_face_names = []
            
            try:
                # --- OPTIMIZATION: Detect from RGB NumPy array, not PIL Image ---
                rgb_process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                
                # 1. Detect ONCE. post_process=False is slightly faster.
                boxes, probs = mtcnn.detect(rgb_process_frame, landmarks=False) # Landmarks=False is faster
                
                if boxes is not None:
                    # 2. EXTRACT from boxes, don't detect again
                    face_tensors = mtcnn.extract(rgb_process_frame, boxes, save_path=None)
                else:
                    face_tensors = None # No faces found

                if face_tensors is not None:
                    face_tensors = face_tensors.to(BATCH_DEVICE)
                    
                    with torch.no_grad():
                        embeddings = resnet(face_tensors)
                    
                    # embeddings_np is shape (N, 512) where N is num faces
                    embeddings_np = embeddings.cpu().numpy()

                    # Check if we have any known faces to compare against
                    if known_encodings_np.size > 0:
                        # 1. Compute all KxN distances at once using broadcasting
                        # (K, 1, 512) - (1, N, 512) -> (K, N, 512) -> (K, N)
                        diff = known_encodings_np[:, np.newaxis, :] - embeddings_np[np.newaxis, :, :]
                        face_distances_matrix = np.linalg.norm(diff, axis=2)

                        # 2. Find best match (smallest distance) for each detected face (N)
                        #    axis=0 finds the minimum value down each column
                        best_match_indices = np.argmin(face_distances_matrix, axis=0) # shape (N,)
                        
                        # 3. Get the distance value for each best match
                        best_match_distances = face_distances_matrix[best_match_indices, np.arange(len(best_match_indices))] # shape (N,)

                        # 4. Check if these distances are within tolerance
                        matches = best_match_distances <= TOLERANCE # shape (N,)
                        
                        # 5. Build the name list
                        current_face_names = []
                        for i, match in enumerate(matches):
                            if match:
                                name = known_names[best_match_indices[i]]
                            else:
                                name = "Unknown"
                            current_face_names.append(name)
                    else:
                        # No known faces, label all as Unknown
                        current_face_names = ["Unknown"] * len(embeddings_np)

                # Store results for skipped frames
                last_boxes = boxes
                last_names = current_face_names

            except (RuntimeError, cv2.error, TypeError) as e:
                # Catch errors (like the 'no len' error, or 'bad frame')
                print(f"Warning: Skipping bad frame. Error: {e}")
                cv2.putText(display_frame, "Error: Bad camera frame", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                last_boxes = None
                last_names = []
        else:
            # Use the results from the last processed frame
            boxes = last_boxes
            current_face_names = last_names
        
        # --- Drawing (runs every frame for smoothness) ---
        if boxes is not None:
            for (left, top, right, bottom), name in zip(boxes, current_face_names):
                
                # Scale the box coordinates back to the original frame size
                top = int(top * scale_y)
                right = int(right * scale_x)
                bottom = int(bottom * scale_y)
                left = int(left * scale_x)

                if name == "Unknown":
                    box_color = (0, 0, 255) # Red
                else:
                    box_color = (0, 255, 0) # Green

                cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
                
                text_y = top - 10 
                box_top = top - text_h - 20 
                box_bottom = top - 5 
                
                if box_top < 0:
                    box_top = top + 5
                    box_bottom = top + text_h + 20
                    text_y = top + text_h + 10 
                
                cv2.rectangle(display_frame, (left, box_top), (left + text_w + 10, box_bottom), box_color, cv2.FILLED)
                cv2.putText(display_frame, name, (left + 5, text_y), font, font_scale, (255, 255, 255), font_thickness)

        cv2.imshow('Face Recognition - Press "q" to quit', display_frame)
        
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Video stream stopped.")

