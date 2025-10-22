import cv2
import os
import face_recognition
import numpy as np
import time
import sys
import signal
from PIL import Image
import subprocess

# ---------------- Settings ----------------
DATASET_PATH = "dataset"
IMAGE_COUNT = 50

# --- FFmpeg Settings ---
# Find camera detial by running: 
# ffmpeg -list_devices true -f dshow -i dummy
# ffmpeg -f dshow -list_options true -i "video=<CAMERA_NAME>"
CAMERA_NAME = "USB2.0 HD UVC WebCam"  # <--- SET THIS TO YOUR CAMERA'S NAME
CAM_WIDTH = 640  # SET CAM PARAM BASED ON YOUR CAMERA
CAM_HEIGHT = 480
CAM_FPS = 20

# --- General Settings ---
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

# --- RESTORED: load Haar cascade (OpenCV fallback) ---
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------ FFmpeg pipe capture class ------------------
class FFMPEGPipeCapture:
    """
    Uses ffmpeg (dshow) to capture raw BGR frames and streams them to Python.
    Provides read(), is_opened(), release() methods.
    """
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

# ------------------ Camera Functions ------------------
def open_camera():
    """Tries to open the camera using FFmpeg."""
    try:
        print(f"Trying FFmpeg pipe for camera '{CAMERA_NAME}'...")
        ffcap = FFMPEGPipeCapture(CAMERA_NAME, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS)
        if ffcap.is_opened():
            print("Successfully opened camera via FFmpeg pipe.")
            return ffcap
        else:
            print("FFmpeg capture .is_opened() returned False.")
            ffcap.release()
            return None
    except Exception as e:
        print(f"Opening FFmpeg pipe failed: {e!r}")
        return None

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
            # We must make a writeable copy to test cvtColor
            _ = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            print("Camera warmup successful. RGB frame acquired.")
            return True
        except (cv2.error, RuntimeError):
            if i % BAD_FRAME_PRINT_STEP == 0:
                print(f"Warmup: unsupported frame type {i+1}/{tries}")
            time.sleep(WARMUP_SLEEP)
            continue
    return False

# ------------------ Main ------------------
cap = open_camera()

if not (cap and cap.is_opened()):
    print("Could not open webcam. Check CAM_NAME, if ffmpeg is on PATH, & if other apps are using the camera.")
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
                cap = open_camera() # Try to reopen
                            
                if not (cap and cap.is_opened() and warmup_camera(cap)):
                    print("Failed to recover camera. Exiting.")
                    break
                consecutive_bad_frames = 0
            time.sleep(0.01)
            continue
        
        # --- FIX: Make frame writeable ---
        frame = frame.copy()

        # normalize: handle 4-channel, non-uint8, single-channel
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
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

        rgb_frame = np.ascontiguousarray(rgb_frame).copy()
        consecutive_bad_frames = 0

        # --- RESTORED: Multi-step Face Detection with Fallback ---
        face_locations = None
        detector_used = None

        # Attempt 1: direct call (Fastest)
        try:
            face_locations = face_recognition.face_locations(rgb_frame)
            detector_used = "face_recognition"
        except Exception as e1:
            # Attempt 2: PIL roundtrip (Handles some memory layout issues)
            try:
                pil = Image.fromarray(rgb_frame)
                arr = np.array(pil)
                arr = np.ascontiguousarray(arr).copy()
                face_locations = face_recognition.face_locations(arr)
                detector_used = "face_recognition (PIL)"
            except Exception as e2:
                # Attempt 3: Haar cascade fallback (Handles non-RGB frames)
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                    if len(faces) > 0:
                        face_locations = []
                        for (x, y, w, h) in faces:
                            top, left, bottom, right = y, x, y + h, x + w
                            face_locations.append((top, right, bottom, left))
                        detector_used = "opencv_haar (fallback)"
                    else:
                        face_locations = []
                        detector_used = "none (haar)"
                except Exception as e3:
                    print(f"All detectors failed. Error: {e3}")
                    face_locations = []
                    detector_used = "none (error)"

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

