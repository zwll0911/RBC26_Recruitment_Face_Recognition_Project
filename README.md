# RBC26_Recruitment_Face_Recognition_Project

## Real-Time Face Recognition Project

This project implements a real-time face recognition pipeline using Python. It captures video via FFmpeg, detects faces using MediaPipe, generates embeddings using FaceNet (via `facenet-pytorch`), and recognizes known faces against a pre-encoded database.

The system is divided into three main scripts:

1.  `capture_faces.py`: Collects face images using FFmpeg and MediaPipe to build a dataset of known individuals.
2.  `encode_faces.py`: Processes the collected images using MTCNN for detection and FaceNet (`InceptionResnetV1`) to create and save embeddings for known faces. Includes batch processing, resizing, and optional deletion of problematic images.
3.  `recognize_faces.py`: Performs real-time face recognition using a live webcam feed (via FFmpeg), detecting faces with MediaPipe, generating live embeddings with FaceNet, and matching against the saved database. Uses threading and frame skipping for optimization.

## How It Works

The project uses a modern deep learning approach for high-accuracy face recognition:

* **Camera Capture**: Uses **FFmpeg** via a subprocess pipe (`FFMPEGPipeCapture` class) for robust camera access (specifically targeting Windows DirectShow). Captures raw BGR frames.
* **Face Detection**:
    * `capture_faces.py` & `recognize_faces.py`: Use **MediaPipe Face Detection** for fast and reliable bounding box detection on CPU/GPU.
    * `encode_faces.py`: Uses **MTCNN** (from `facenet-pytorch`) for potentially higher accuracy detection during the offline encoding phase, with batch processing and fallback.
* **Face Embedding**: A pre-trained **InceptionResnetV1 (FaceNet)** model from `facenet-pytorch` (trained on VGGFace2) computes a 512-dimension vector (embedding) for each detected face.
* **Encoding**: The `encode_faces.py` script iterates through the `dataset` folder, detects faces, generates embeddings, and saves them along with corresponding names into a `.pickle` file (`encodings_facenet.pickle`).
* **Recognition**: The `recognize_faces.py` script loads the saved embeddings. In real-time, it detects faces, computes their live embeddings, and calculates the Euclidean distance to all known embeddings. If the minimum distance is below a set `RECOGNITION_THRESHOLD`, the corresponding name is assigned; otherwise, it's labeled "Unknown".
* **Optimization**: The recognition script uses **threading** to decouple camera capture from processing and processes only every Nth frame (`FRAME_PROCESS_INTERVAL`) to prevent accumulating lag. Embeddings for multiple faces in a frame are generated in batches.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/zwll0911/RBC26_Recruitment_Face_Recognition_Project.git
    cd RBC26_Recruitment_Face_Recognition_Project
    ```

2.  **Install FFmpeg** (Required for Camera Capture)
    * **Windows (using Winget):**
        ```bash
        winget install --id=Gyan.FFmpeg -e
        ```
    * **Other OS / Manual:** Download from the [FFmpeg website](https://ffmpeg.org/download.html) and ensure the `ffmpeg` executable is in your system's PATH.

3.  **Create a Virtual Environment** (Recommended)
    * Using Python 3.9+ is generally recommended for recent versions of these libraries (here we use python3.11.9).
    ```bash
    py -3.11 -m venv face_rec
    ```
    * Activate the environment:
        * **Windows (PowerShell):**
            ```powershell
            # You might need to allow script execution first (run PowerShell as Administrator)
            # Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
            .\face_rec\Scripts\activate
            ```
        * **Windows (Command Prompt):**
            ```cmd
            .\face_rec\Scripts\activate.bat
            ```
        * **macOS/Linux:**
            ```bash
            source face_rec/bin/activate
            ```

4.  **Install Python Dependencies**
    * All other required libraries are listed in `requirements.txt`. Install them using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    > * **(GPU Support):** If you have a CUDA-enabled NVIDIA GPU, ensure you have the correct NVIDIA drivers and CUDA Toolkit installed. Then, install the GPU version of PyTorch by following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/). This will significantly speed up encoding and recognition.

## Usage (Step-by-Step)

Follow these steps in order to run the project.

### Step 1: Find Your Camera Name (Important!)

```bash
ffmpeg -list_devices true -f dshow -i dummy
```

- Look for your webcam under "DirectShow video devices". Common names include "Integrated Camera", "USB Video Device", or the specific model name.

```bash
# Replace <CAMERA_NAME> with the exact name found above
ffmpeg -f dshow -list_options true -i video="<CAMERA_NAME>"
```

- Update the CAMERA_DSHOW_NAME variable at the top of capture_faces.py and recognize_faces.py with the exact name found. You might also want to adjust CAM_WIDTH, CAM_HEIGHT, and CAM_FPS based on your camera's capabilities and desired performance.

### Step 2: Capture Faces (capture_faces.py)

Build your dataset of known faces.

```bash
python capture_faces.py
```

- Enter a name for the person when prompted.
- A folder `dataset/<name>` will be created.
- A window will show the camera feed. Position your face.
- Press `s` to save the detected face crop. The script uses MediaPipe to detect faces and saves the cropped BGR image.
- Collect multiple images (e.g., 20-50) per person with slight variations.
- Press `q` to quit capturing for the current person.
- Run the script again for each new person.

### Step 3: Encode Faces (encode_faces.py)

Process the captured images to create embeddings.

```bash
python encode_faces.py
```

- Scans the `dataset/` directory.
- Resizes images, uses MTCNN to detect faces (with fallback for tricky batches), computes FaceNet embeddings.
- Optionally deletes images that cause errors or where no face is detected (controlled by `DELETE_FAILED_IMAGES` flag in the script).
- Saves embeddings and names to `encodings_facenet.pickle`.
- Run this whenever you add/change images in the `dataset` folder.

### Step 4: Recognize Faces (recognize_faces.py)

Run the real-time recognition.

```bash
python recognize_faces.py
```

- Loads `encodings_facenet.pickle`.
- Starts the camera feed via FFmpeg (using threading).
- Detects faces (MediaPipe), generates embeddings (FaceNet), and compares them every few frames (`FRAME_PROCESS_INTERVAL`).
- Draws boxes and labels (Name or "Unknown" + distance) on the video feed.
- Press `q` to quit.

### Core Dependencies

- `FFmpeg`: External tool for robust camera capture.
- `opencv-python`: For displaying video, drawing shapes, saving images (imwrite), and image format conversions.
- `mediapipe`: For fast face detection in the capture and recognition scripts.
- `torch` & `torchvision`: Core deep learning framework.
- `facenet-pytorch`: Provides pre-trained MTCNN (detector for encoding) and InceptionResnetV1 (FaceNet embedder).
- `numpy`: For numerical operations, especially on image arrays and embeddings.
- `Pillow` (PIL): Used for image loading and resizing during encoding.
- `tqdm`: Displays progress bars during encoding.
