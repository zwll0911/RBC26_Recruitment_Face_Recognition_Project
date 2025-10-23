import subprocess
import os
import time
import numpy as np
import mediapipe as mp
import cv2
import datetime

# --- Configuration ---
# Find camera detials by running: 
# ffmpeg -list_devices true -f dshow -i dummy
# ffmpeg -f dshow -list_options true -i "video=<CAMERA_NAME>"
CAMERA_DSHOW_NAME = "USB2.0 HD UVC WebCam"
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
WARMUP_FRAMES = 15
SAVE_DIR_BASE = "dataset"

# --- FFMPEGPipeCapture Class ---
class FFMPEGPipeCapture:
    def __init__(self, dshow_name, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
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

        # Check if ffmpeg started correctly after a short delay
        time.sleep(0.5)
        if self.proc.poll() is not None:
              stderr_output = self.proc.stderr.read().decode('utf-8', errors='ignore')
              if "Could not find video device" in stderr_output or "error opening device" in stderr_output:
                   raise RuntimeError(f"ffmpeg could not find or open the DirectShow device named '{dshow_name}'. "
                                      f"Please check camera connections and the exact device name. Error: {stderr_output}")
              else:
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
            except Exception as e:
                print(f"Error reading from ffmpeg stdout pipe: {e}")
                self._alive = False
                return False, None

            if not chunk:
                poll_result = self.proc.poll()
                stderr_output = self.proc.stderr.read().decode('utf-8', errors='ignore')
                print(f"ffmpeg process stdout pipe closed unexpectedly. Poll result: {poll_result}. Stderr: {stderr_output}")
                self._alive = False
                return False, None

            buf += chunk

            if timeout is not None and (time.time() - t0) > timeout:
                print(f"ffmpeg read timeout after {timeout}s ({len(buf)}/{bytes_to_read} bytes read)")
                try:
                    stderr_output = self.proc.stderr.read(1024).decode('utf-8', errors='ignore')
                    if stderr_output:
                         print(f"ffmpeg stderr during timeout: {stderr_output}")
                except Exception:
                     pass
                return False, None

        frame = np.frombuffer(buf, dtype=np.uint8)
        try:
            frame = frame.reshape((self.height, self.width, 3))
        except ValueError as e:
            print(f"Error reshaping frame buffer ({len(buf)} bytes read, expected {self.frame_size}): {e}")
            return False, None
        except Exception as e:
            print(f"Unexpected error reshaping frame buffer: {e}")
            return False, None

        return True, frame

    def is_opened(self):
        if self._alive and self.proc and self.proc.poll() is None:
            return True
        if self._alive:
             print(f"ffmpeg process seems to have exited (poll result: {self.proc.poll()}).")
             self._alive = False
        return False

    def release(self):
        print("Releasing ffmpeg capture...")
        if self.proc is not None:
            if self.proc.poll() is None:
                 try:
                     print("Terminating ffmpeg process...")
                     self.proc.terminate()
                     try:
                         self.proc.wait(timeout=1.0)
                         print("ffmpeg process terminated.")
                     except subprocess.TimeoutExpired:
                         print("ffmpeg did not terminate gracefully, killing...")
                         self.proc.kill()
                         self.proc.wait(timeout=1.0)
                         print("ffmpeg process killed.")
                 except Exception as e:
                     print(f"Error during ffmpeg terminate/kill: {e}")
            else:
                 print("ffmpeg process was already terminated.")

            try:
                 if self.proc.stdout: self.proc.stdout.close()
                 if self.proc.stderr: self.proc.stderr.close()
            except Exception as e:
                 print(f"Error closing ffmpeg pipes: {e}")

        self.proc = None
        self._alive = False
        print("ffmpeg capture released.")
# --------------------------------------------------------

# --- Main Face Detection Loop ---
if __name__ == "__main__":

    user_name = input("Enter the name for saving faces: ").strip()
    if not user_name:
        print("Name cannot be empty. Exiting.")
        exit()

    save_path = os.path.join(SAVE_DIR_BASE, user_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving detected faces to: {save_path}")
    # ----------------------------------------------------

    ff_cap = None
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )

    try:
        ff_cap = FFMPEGPipeCapture(CAMERA_DSHOW_NAME, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
        if not ff_cap.is_opened():
             raise RuntimeError(f"Could not open camera '{CAMERA_DSHOW_NAME}' via FFmpeg.")

        print(f"Warming up camera for {WARMUP_FRAMES} frames...")
        for _ in range(WARMUP_FRAMES):
            ret, _ = ff_cap.read(timeout=2.0)
            if not ret:
                raise RuntimeError("Failed to read frame during camera warmup.")
        print("Warmup complete.")

        print("\nStarting face detection loop.")
        print("Press 'S' in the window to save detected face(s).")
        print("Press 'Q' in the window to stop.")
        frame_count = 0
        saved_face_count = 0
        start_time = time.time()

        while ff_cap.is_opened():
            ret, frame_bgr_readonly = ff_cap.read(timeout=5.0)
            if not ret or frame_bgr_readonly is None:
                print("Failed to read frame from camera. Exiting loop.")
                break

            # Create a writable copy for drawing
            frame_bgr = frame_bgr_readonly.copy()

            # Convert BGR to RGB and ensure contiguous for MediaPipe
            frame_rgb_non_contig = frame_bgr[:, :, ::-1]
            frame_rgb = np.ascontiguousarray(frame_rgb_non_contig, dtype=np.uint8)

            # MediaPipe Detection
            frame_rgb.flags.writeable = False
            results = face_detection.process(frame_rgb)

            frame_count += 1
            current_face_locations = [] # Store locations found in *this* frame

            # Process detections and get coordinates
            if results.detections:
                ih, iw, _ = frame_bgr.shape # Get dimensions once
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    if not hasattr(bboxC, 'xmin') or bboxC.xmin is None: continue

                    xmin = int(bboxC.xmin * iw)
                    ymin = int(bboxC.ymin * ih)
                    width = int(bboxC.width * iw)
                    height = int(bboxC.height * ih)

                    # Ensure coordinates are within bounds and add padding if needed (optional)
                    padding = 0 # Adjust padding if want slightly larger crops
                    xmin = max(0, xmin - padding)
                    ymin = max(0, ymin - padding)
                    right = min(iw, xmin + width + padding * 2)
                    bottom = min(ih, ymin + height + padding * 2)
                    left = xmin
                    top = ymin

                    if right > left and bottom > top:
                         face_coords = (top, right, bottom, left)
                         current_face_locations.append(face_coords)
                         cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display the WRITABLE frame_bgr with boxes
            cv2.imshow('Camera Feed (S=Save, Q=Quit)', frame_bgr)

            # --- Check for key presses ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quit key pressed.")
                break

            elif key == ord('s'):
                if current_face_locations:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    num_saved_this_frame = 0
                    for idx, (top, right, bottom, left) in enumerate(current_face_locations):
                        # Crop the face from the BGR frame (the one we drew on is fine)
                        face_crop = frame_bgr[top:bottom, left:right]

                        if face_crop.size == 0:
                             print(f"  Skipping empty crop for face {idx+1}...")
                             continue

                        # Create filename and save
                        filename = f"{user_name}_{timestamp}_{idx+1}.jpg"
                        filepath = os.path.join(save_path, filename)
                        try:
                            cv2.imwrite(filepath, face_crop)
                            print(f"  Saved: {filepath}")
                            num_saved_this_frame += 1
                        except Exception as write_error:
                            print(f"  Error saving {filepath}: {write_error}")

                    if num_saved_this_frame > 0:
                        saved_face_count += num_saved_this_frame
                        print(f"Saved {num_saved_this_frame} face(s) this frame. Total saved: {saved_face_count}")
                    else:
                        print("Detected faces but failed to save any crops.")
                else:
                    print("Save ('s') pressed, but no faces detected in the current frame.")
            # --------------------------------

            # Calculate FPS
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

    except KeyboardInterrupt:
        print("\nStopping detection loop (Ctrl+C)...")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Release resources
        if 'face_detection' in locals() and face_detection:
             face_detection.close()
        if ff_cap is not None:
            ff_cap.release()
        cv2.destroyAllWindows()
        print("OpenCV windows closed.")
