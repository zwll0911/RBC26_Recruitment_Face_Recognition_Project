import face_recognition
import cv2
import pickle
import imutils # For resizing frames
import numpy as np
import time
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

print("Starting video face recognition...")

# --- Settings ---
ENCODINGS_FILE = "encodings_facenet.pickle" # *** LOADS THE FACENET FILE ***
RESIZE_WIDTH = 500 # Resize video frame for faster processing
TOLERANCE = 0.6 # FaceNet often needs a slightly higher tolerance
CAMERA_INDEX = 0 # Match the index from create_dataset.py

# --- Load Encodings ---
print(f"Loading known face encodings from {ENCODINGS_FILE}...")
try:
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    
    # *** CRITICAL: Convert list of lists back to numpy array ***
    data["encodings"] = np.array(data["encodings"])
    
except FileNotFoundError:
    print(f"Error: Encodings file not found: {ENCODINGS_FILE}")
    print("Please run your 'encode_faces.py' first to create the file.")
    exit()

# --- Start Webcam ---
# *** Kept all robust camera logic from create_dataset.py ***
print(f"Trying to open camera at index: {CAMERA_INDEX} (with DSHOW flag)")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Kept Smart Camera Warm-up ---
print("Warming up camera... (This may take a few seconds)")
warmup_frames_to_try = 200
warmup_success = False
for i in range(warmup_frames_to_try):
    ret, frame = cap.read()
    if not ret:
        print("Error during camera warm-up (failed to read frame).")
        cap.release()
        exit()
    
    try:
        # Try to convert. If this works, the camera is ready.
        # *** FIXED TYPO HERE ***
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        warmup_success = True
        print("Camera warmup successful. RGB frame acquired.")
        break # Exit the warmup loop
    except (RuntimeError, cv2.error):
        # This is an expected "bad frame"
        if i % 20 == 0:
            print(f"Warmup... discarding bad frame {i+1}/{warmup_frames_to_try}")
        time.sleep(0.05)

if not warmup_success:
    print("\n--- CRITICAL ERROR ---")
    print(f"Failed to get a valid RGB frame from the camera after {warmup_frames_to_try} attempts.")
    print("Please check camera drivers or try disabling 3D camera in Device Manager.")
    print("----------------------")
    cap.release()
    exit()

# --- *** NEW: Load FaceNet Models *** ---
print("Loading FaceNet models...")
BATCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {BATCH_DEVICE}")
mtcnn = MTCNN(
    keep_all=True, # <-- IMPORTANT: Find all faces in the frame
    device=BATCH_DEVICE
)
# *** FIXED TYPO HERE ***
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(BATCH_DEVICE)
print("FaceNet models loaded.")

print("Webcam started. Starting recognition...")

# --- Main Loop ---
while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("Warning: Failed to capture frame.")
        continue

    # Create a copy to draw on
    display_frame = frame.copy()

    # --- Create a smaller frame for processing ---
    # Resize frame for faster processing
    process_frame = imutils.resize(frame, width=RESIZE_WIDTH)
    
    # Calculate scale ratio for drawing boxes later
    scale_y = frame.shape[0] / process_frame.shape[0]
    scale_x = frame.shape[1] / process_frame.shape[1]

    # --- *** NEW: FaceNet Detection & Recognition *** ---
    boxes = None
    current_face_names = []
    
    try:
        # 1. Convert frame from BGR (OpenCV) to RGB (PIL)
        # *** FIXED TYPO HERE ***
        img_pil = Image.fromarray(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))
        
        # 2. Get boxes for drawing
        #    Note: This runs detection
        boxes, probs = mtcnn.detect(img_pil)
        
        # 3. Get cropped face tensors for encoding
        #    Note: This runs detection *again* but returns cropped tensors
        face_tensors = mtcnn(img_pil, save_path=None)

        if face_tensors is not None:
            # Move tensors to the correct device
            face_tensors = face_tensors.to(BATCH_DEVICE)
            
            # 4. Get all embeddings in a batch
            with torch.no_grad():
                embeddings = resnet(face_tensors)
            
            # 5. Convert to numpy
            embeddings_np = embeddings.cpu().numpy()

            # 6. Compare each detected face
            for emb_np in embeddings_np:
                # 'emb_np' is now a (512,) numpy array, matching our file
                matches = face_recognition.compare_faces(
                    data["encodings"], emb_np, tolerance=TOLERANCE
                )
                
                name = "Unknown" # Default name

                # --- Find Best Match ---
                face_distances = face_recognition.face_distance(data["encodings"], emb_np)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    # If the best match is a 'True' match (within tolerance)
                    name = data["names"][best_match_index]

                current_face_names.append(name)

    except (RuntimeError, cv2.error, TypeError) as e:
        print(f"Warning: Skipping bad frame. Error: {e}")
        # Draw an error on the screen
        cv2.putText(display_frame, "Error: Bad camera frame", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        pass # Show the frame without boxes

    # --- Draw Results on Frame ---
    # We draw on the original, full-size 'display_frame'
    
    # Check if boxes is not None before zipping
    if boxes is not None:
        for (left, top, right, bottom), name in zip(boxes, current_face_names):
            
            # --- MTCNN boxes are (L, T, R, B) and float ---
            
            # Scale the box coordinates back to the original frame size
            top = int(top * scale_y)
            right = int(right * scale_x)
            bottom = int(bottom * scale_y)
            left = int(left * scale_x)

            # Set box color
            if name == "Unknown":
                box_color = (0, 0, 255) # Red
            else:
                box_color = (0, 255, 0) # Green

            # Draw a box around the face
            cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)
            
            # --- *** NEW, NICER LABEL CODE *** ---
            
            # 1. Set up font
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            
            # 2. Get the size of the text
            (text_w, text_h), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
            
            # 3. Decide on label position
            # We want it just above the box, at (left, top - 10)
            # But if that goes off-screen, we put it inside at (left, top + 10)
            
            # y-coordinate for the *text*
            text_y = top - 10 
            
            # Coordinates for the *filled box*
            box_top = top - text_h - 20 # 10px padding above text
            box_bottom = top - 5 # 5px padding below text
            
            # If the label would go off-screen (box_top < 0)
            if box_top < 0:
                # Place it just inside the box
                box_top = top + 5
                box_bottom = top + text_h + 20
                text_y = top + text_h + 10 # y position of text baseline
            
            # Draw the filled rectangle background
            cv2.rectangle(display_frame, (left, box_top), (left + text_w + 10, box_bottom), box_color, cv2.FILLED)
            
            # Draw the text on top
            cv2.putText(display_frame, name, (left + 5, text_y), font, font_scale, (255, 255, 255), font_thickness)
            # --- *** END OF NEW LABEL CODE *** ---


    # --- Display the final frame ---
    # This now runs EVERY loop, so you will always see the video feed
    cv2.imshow('Face Recognition - Press "q" to quit', display_frame)

    # Quit logic
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Video stream stopped.")

