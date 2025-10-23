import subprocess
import os
import time
import numpy as np
import mediapipe as mp
import cv2
import pickle
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import threading

# --- Configuration ---
# Find camera detials by running: 
# ffmpeg -list_devices true -f dshow -i dummy
# ffmpeg -f dshow -list_options true -i "video=<CAMERA_NAME>"
CAMERA_DSHOW_NAME = "USB2.0 HD UVC WebCam"
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
WARMUP_FRAMES = 15
ENCODING_FILE = "encodings_facenet.pickle"
RECOGNITION_THRESHOLD = 0.7
BATCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_DIM_FACENET = (160, 160)
FRAME_PROCESS_INTERVAL = 4
# -------------------------------

print(f"Using device: {BATCH_DEVICE}")
print(f"Processing every {FRAME_PROCESS_INTERVAL} frames.")


# --- FFMPEGPipeCapture Class ---
class FFMPEGPipeCapture:
    def __init__(self, dshow_name, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
        self.width=int(width);self.height=int(height);self.frame_size=self.width*self.height*3;self._alive=False;self.proc=None
        cmd=["ffmpeg","-hide_banner","-loglevel","error","-f","dshow","-framerate",str(fps),"-video_size",f"{self.width}x{self.height}","-i",f"video={dshow_name}","-pix_fmt","bgr24","-vcodec","rawvideo","-an","-sn","-f","rawvideo","-"]
        popen_kwargs={"stdin":subprocess.DEVNULL,"stdout":subprocess.PIPE,"stderr":subprocess.PIPE}
        if os.name=="nt":popen_kwargs["creationflags"]=subprocess.CREATE_NO_WINDOW
        try:self.proc=subprocess.Popen(cmd,**popen_kwargs)
        except FileNotFoundError:raise RuntimeError("ffmpeg not found.")
        except Exception as e:raise RuntimeError(f"Failed start: {e!r}")
        time.sleep(0.5)
        if self.proc.poll() is not None:
            stderr_output=self.proc.stderr.read().decode('utf-8',errors='ignore')
            if"Could not find video device"in stderr_output or"error opening device"in stderr_output:raise RuntimeError(f"ffmpeg bad device '{dshow_name}'. Err: {stderr_output}")
            else:raise RuntimeError(f"ffmpeg startup fail. Err: {stderr_output}")
        self._alive=True;print(f"ffmpeg pipe started for '{dshow_name}'")
    def read(self,timeout=2.0):
        if not self.is_opened():return False,None
        buf=b"";bytes_to_read=self.frame_size;t0=time.time()
        while len(buf)<bytes_to_read:
            try:chunk=self.proc.stdout.read(bytes_to_read-len(buf))
            except Exception as e:print(f"Read pipe err: {e}");self._alive=False;return False,None
            if not chunk:
                poll_result=self.proc.poll();stderr_output=self.proc.stderr.read().decode('utf-8',errors='ignore')
                print(f"Pipe closed. Poll:{poll_result}. Stderr:{stderr_output}");self._alive=False;return False,None
            buf+=chunk
            if timeout is not None and(time.time()-t0)>timeout:
                print(f"Read timeout ({len(buf)}/{bytes_to_read} bytes)")
                try:
                    stderr_output=self.proc.stderr.read(1024).decode('utf-8',errors='ignore')
                    if stderr_output:print(f"ffmpeg stderr: {stderr_output}")
                except Exception:pass
                return False,None
        frame=np.frombuffer(buf,dtype=np.uint8)
        try:frame=frame.reshape((self.height,self.width,3))
        except ValueError as e:print(f"Reshape err({len(buf)} bytes): {e}");return False,None
        except Exception as e:print(f"Unexpected reshape err: {e}");return False,None
        return True,frame
    def is_opened(self):
        if self._alive and self.proc and self.proc.poll() is None:return True
        if self._alive:print(f"ffmpeg exited (poll:{self.proc.poll()}).");self._alive=False
        return False
    def release(self):
        print("Releasing ffmpeg...")
        if self.proc is not None:
            if self.proc.poll()is None:
                try:
                    print("Terminating ffmpeg...")
                    self.proc.terminate()
                    try:self.proc.wait(timeout=1.0);print("Terminated.")
                    except subprocess.TimeoutExpired:
                        print("Killing...");self.proc.kill()
                        try:self.proc.wait(timeout=1.0);print("Killed.")
                        except subprocess.TimeoutExpired:print("Kill timed out.")
                        except Exception as we:print(f"Wait kill err: {we}")
                except Exception as te:print(f"Term/kill err: {te}")
            else:print("ffmpeg already terminated.")
            try:
                if self.proc.stdout:self.proc.stdout.close()
                if self.proc.stderr:self.proc.stderr.close()
                print("ffmpeg pipes closed.")
            except Exception as e:print(f"Pipe close err: {e}")
        self.proc=None;self._alive=False;print("ffmpeg capture released.")
# --------------------------------------------------------

# --- Helper: Preprocess face crop for FaceNet ---
def preprocess_face_tensor(face_crop_pil, device):
    if face_crop_pil is None: return None
    try:
        resized = face_crop_pil.resize(RESIZE_DIM_FACENET, Image.Resampling.LANCZOS)
        face_np = np.array(resized, dtype=np.float32) / 255.0
        face_np = (face_np - 0.5) * 2.0 # Normalize [-1, 1]
        face_tensor = torch.as_tensor(face_np).permute(2, 0, 1) # HWC to CHW
        return face_tensor # Keep on CPU initially for batching
    except Exception as e: print(f"Error preprocessing face: {e}"); return None

# --- Threading Globals ---
latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()
# -----------------------------

# --- Capture Thread Function ---
def capture_thread_func(pipe_cap):
    global latest_frame
    print("Capture thread started.")
    while not stop_event.is_set():
        if not pipe_cap.is_opened():
            print("Capture device closed in thread.")
            break
        # --- Increased read timeout ---
        ret, frame = pipe_cap.read(timeout=5.0)
        # ----------------------------
        if ret and frame is not None:
            with frame_lock:
                latest_frame = frame # Update shared frame
        elif pipe_cap.is_opened():
            time.sleep(0.01) # Avoid busy-waiting if read fails temporarily
        else: # Pipe closed
             break # Exit thread if pipe is closed
    print("Capture thread finished.")
# ---------------------------------


# --- Main Face Recognition Loop ---
if __name__ == "__main__":
    ff_cap = None
    capture_thread = None # Thread object
    known_face_encodings = []
    known_face_names = []
    last_processed_frame_data = [] # Store data from last processed frame
    frame_counter_process = 0 # Counter for processing interval

    # --- Load Known Encodings ---
    try:
        print(f"Loading known face encodings from {ENCODING_FILE}...")
        with open(ENCODING_FILE, "rb") as f: data = pickle.load(f)
        known_face_encodings = np.array(data["encodings"])
        known_face_names = data["names"]
        if len(known_face_encodings) == 0: raise ValueError("No encodings found.")
        print(f"Loaded {len(known_face_encodings)} encodings for {len(set(known_face_names))} people.")
    except FileNotFoundError: print(f"Error: Encoding file '{ENCODING_FILE}' not found."); exit()
    except Exception as e: print(f"Error loading encoding file: {e}"); exit()
    # ----------------------------

    # --- Initialize Models ---
    print("Initializing MediaPipe Face Detection...")
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    print(f"Initializing InceptionResnetV1 (FaceNet) on {BATCH_DEVICE}...")
    try: resnet = InceptionResnetV1(pretrained='vggface2').eval().to(BATCH_DEVICE)
    except Exception as e: print(f"Error initializing FaceNet: {e}"); exit()
    # --------------------------

    try:
        # 1. Initialize FFmpeg capture
        ff_cap = FFMPEGPipeCapture(CAMERA_DSHOW_NAME, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
        if not ff_cap.is_opened(): raise RuntimeError(f"Could not open camera '{CAMERA_DSHOW_NAME}'.")

        # --- Start Capture Thread ---
        print("Starting capture thread...")
        capture_thread = threading.Thread(target=capture_thread_func, args=(ff_cap,), daemon=True)
        capture_thread.start()
        # --------------------------

        # --- Camera Warmup (using shared frame) ---
        print(f"Warming up camera (waiting for first frame)...")
        warmup_start = time.time()
        while time.time() - warmup_start < 5.0: # Wait up to 5 seconds for first frame
             with frame_lock:
                  if latest_frame is not None:
                       print("First frame received. Warmup complete.")
                       break
             time.sleep(0.1) # Wait a bit before checking again
        else: # If loop finishes without break
             raise RuntimeError("Failed to get first frame from capture thread during warmup.")
        # --------------------

        print("\nStarting face recognition loop. Press 'q' in window to stop.")
        fps_frame_count = 0
        start_time = time.time()

        # --- Main Processing Loop ---
        while not stop_event.is_set():
            # 2. Get the latest frame from the capture thread
            current_frame_bgr = None
            with frame_lock:
                if latest_frame is not None:
                    current_frame_bgr = latest_frame.copy()

            if current_frame_bgr is None:
                # If capture thread hasn't provided a frame yet, wait briefly
                time.sleep(0.01)
                continue

            frame_counter_process += 1

            # --- Process only every Nth frame ---
            if frame_counter_process % FRAME_PROCESS_INTERVAL == 0:
                last_processed_frame_data = []
                preprocessed_faces = []
                boxes_for_drawing = []

                # Convert to RGB and make contiguous
                frame_rgb_non_contig = current_frame_bgr[:, :, ::-1]
                frame_rgb = np.ascontiguousarray(frame_rgb_non_contig, dtype=np.uint8)

                # 3. Detect Faces (MediaPipe)
                frame_rgb.flags.writeable = False
                results = face_detection.process(frame_rgb)

                # 4. Preprocess all detected faces for batching
                if results.detections:
                    ih, iw, _ = frame_rgb.shape
                    for detection in results.detections:
                        bboxC=detection.location_data.relative_bounding_box
                        if not hasattr(bboxC,'xmin')or bboxC.xmin is None:continue
                        xmin=max(0,int(bboxC.xmin*iw));ymin=max(0,int(bboxC.ymin*ih))
                        width=int(bboxC.width*iw);height=int(bboxC.height*ih)
                        right=min(iw,xmin+width);bottom=min(ih,ymin+height)
                        left=xmin;top=ymin
                        box=(top,right,bottom,left)
                        if right<=left or bottom<=top:continue
                        face_crop_np=frame_rgb[top:bottom,left:right]
                        if face_crop_np.size==0:continue
                        face_crop_pil=Image.fromarray(face_crop_np)
                        face_tensor=preprocess_face_tensor(face_crop_pil,"cpu")
                        if face_tensor is not None:
                            preprocessed_faces.append(face_tensor)
                            boxes_for_drawing.append(box)


                # --- 5. Batch Embedding Generation and Comparison ---
                if preprocessed_faces:
                    try:
                        batch_face_tensors=torch.stack(preprocessed_faces).to(BATCH_DEVICE)
                        live_embeddings_batch=None
                        with torch.no_grad():
                            live_embeddings_batch_tensor=resnet(batch_face_tensors)
                            live_embeddings_batch=live_embeddings_batch_tensor.cpu().numpy()
                        if live_embeddings_batch is not None:
                            for i in range(len(live_embeddings_batch)):
                                live_encoding=live_embeddings_batch[i]
                                current_box=boxes_for_drawing[i]
                                name="Unknown";min_dist=RECOGNITION_THRESHOLD
                                distances=np.linalg.norm(known_face_encodings-live_encoding,axis=1)
                                best_match_index=np.argmin(distances)
                                min_dist=distances[best_match_index]
                                if min_dist<RECOGNITION_THRESHOLD:
                                    name=known_face_names[best_match_index]
                                last_processed_frame_data.append((current_box,name,min_dist))
                    except Exception as batch_e:
                        print(f"Batch embed/compare err: {batch_e}")
                        last_processed_frame_data=[] # Clear results on error

            # --- End Frame Processing Block ---

            # 6. Draw Results (Always draw using the *last processed* data on the current frame)
            display_frame = current_frame_bgr # Draw on the frame grabbed for this loop iteration
            for (top, right, bottom, left), name, dist in last_processed_frame_data:
                label = f"{name} ({dist:.2f})"
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(display_frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 7. Display Frame (Always display)
            cv2.imshow('Face Recognition (Q to Quit)', display_frame)
            fps_frame_count += 1

            # 8. Check for Quit Key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit key pressed.")
                stop_event.set() # Signal capture thread to stop
                break

            # Calculate Display FPS
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = fps_frame_count / elapsed_time
                print(f"Display FPS: {fps:.2f}")
                fps_frame_count = 0
                start_time = time.time()

    except KeyboardInterrupt: print("\nStopping loop (Ctrl+C)..."); stop_event.set()
    except RuntimeError as e: print(f"Runtime Error: {e}"); stop_event.set()
    except Exception as e: print(f"An unexpected error occurred: {e}"); stop_event.set()
    finally:
        # --- Clean up ---
        if stop_event: stop_event.set() # Ensure event is set
        if capture_thread:
             print("Waiting for capture thread to finish...")
             capture_thread.join(timeout=2.0) # Wait for thread to exit
             if capture_thread.is_alive():
                  print("Capture thread did not finish cleanly.")
        if 'face_detection' in locals() and face_detection: face_detection.close()
        if ff_cap is not None: ff_cap.release() # Release ffmpeg pipe AFTER thread finishes
        cv2.destroyAllWindows()
        print("Resources released.")
