# RBC26_Recruitment_Face_Recognition_Project

## Real-Time Face Recognition Project

This project implements a complete real-time face recognition pipeline using Python, OpenCV, FaceNet, and FFmpeg. The system is broken into three main scripts:

1.  `capture_faces.py`: Collects and stores face images using FFmpeg (or OpenCV fallback) to build a dataset.
2.  `encode_faces.py`: Processes the collected images using FaceNet (via `facenet-pytorch`) to create known face embeddings.
3.  `recognize_faces.py`: Performs real-time face recognition using a live webcam feed captured via FFmpeg, leveraging FaceNet for detection and recognition, with optimizations like frame skipping.

## How It Works

The project uses a modern deep learning approach for high-accuracy face recognition:

* **Camera Capture**: It primarily uses **FFmpeg** via a subprocess pipe for robust camera access on Windows (DirectShow). If FFmpeg fails, it falls back to using **OpenCV** `VideoCapture`. Includes logic for camera warmup and automatic reopening on errors.
* **Face Detection**: It uses a **Multi-Task Cascaded Neural Network (MTCNN)** from the `facenet-pytorch` library to accurately detect faces in images (`encode_faces.py`) and video frames (`recognize_faces.py`). `capture_faces.py` uses the `face_recognition` library's detector with an OpenCV Haar Cascade fallback for dataset creation.
* **Face Embedding**: For each detected face, it uses a pre-trained **InceptionResnetV1 (FaceNet)** model (via `facenet-pytorch`) to compute a 512-dimension vector (an "embedding") that numerically represents the face.
* **Face Matching**: The recognition script compares the embeddings of faces seen on the webcam against a pre-computed database of "known" embeddings using efficient vectorized NumPy calculations. A distance calculation (Euclidean distance with a tolerance) determines if the face is a match or "Unknown".

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/zwll0911/RBC26_Recruitment_Face_Recognition_Project.git
    cd RBC26_Recruitment_Face_Recognition_Project
    ```

2.  **Install FFmpeg** (Required for Primary Camera Capture)
    ```bash
    winget install --id=Gyan.FFmpeg -e
    ```

4.  **Create a Virtual Environment** (Recommended)
    * Using Python 3.11 is suggested as the provided `dlib` wheel is for this version.
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

5.  **Install the pre-compiled package for `dlib`**
    * The `face-recognition` library depends on `dlib`, which can sometimes be tricky to install directly. Using the provided wheel file for Windows/Python 3.11 is the easiest way if it matches your system. Make sure the `.whl` file is in your project directory.
    ```bash
    python -m pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
    ```
    > **Note:** If you are not on 64-bit Windows with Python 3.11, you might need to find a different `dlib` wheel file or try installing `dlib` directly (`pip install dlib`), which may require installing CMake and a C++ compiler first.

6.  **Install Other Dependencies**
    * All other required libraries are listed in `requirements.txt`. Install them using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    > **GPU Support:** If you have a CUDA-enabled NVIDIA GPU and the correct NVIDIA drivers installed, `torch` should automatically install with GPU support. This significantly speeds up the face detection and embedding process. If not, it will default to using the CPU.

## Usage (Step-by-Step)

Follow these steps in order to run the project.

### Step 1: Capture Faces

First, build a dataset of known faces. Run the `capture_faces.py` script.

```bash
python capture_faces.py
```

- You will be prompted to enter a name for the person.
- A new folder with this name will be created in the `dataset/` directory.
- The script will attempt to open the webcam specified by `CAMERA_NAME` using FFmpeg. If it fails, it will try the OpenCV fallback.
- A webcam window will open after warmup. Position your face in the frame.
- Press the `s` key to save a frame. The script will only save when exactly one face is detected (using `face_recognition` or Haar cascade).
- Collect around 50 images for good results, varying angles and lighting slightly.
- Press `q` to quit when finished.

Repeat this process for each person you want to add to the dataset.

### Step 2: Encode Faces

Next, process the captured images into FaceNet embeddings.

```bash
python encode_faces.py
```

- This script will scan the `dataset/` directory.
- For each image, it uses MTCNN to detect the face, then computes its 512-d FaceNet embedding using `InceptionResnetV1`.
- All embeddings and their corresponding names are saved to `encodings_facenet.pickle`.
- You only need to run this script once after you've added or changed images in the `dataset` folder.

### Step 3: Recognize Faces

Finally, run the real-time recognition script.

```bash
python recognize_faces.py
```

- This will load the `encodings_facenet.pickle` file.
- It will open your webcam using the FFmpeg method (with OpenCV fallback).
- It detects faces using MTCNN and computes FaceNet embeddings every few frames (`FRAME_SKIP` setting).
- For each detected face, it computes its embedding and compares it to all known embeddings using fast NumPy calculations.
- A green box (for known faces) or red box (for "Unknown" faces) will be drawn, along with a name label. Skipped frames reuse the previous detection results for smooth display.
- Press `q` to quit the video stream.

### Core Dependencies

- `opencv-python`: For image processing, drawing, fallback camera access, and Haar cascade.
- `face-recognition`: Used for face detection (in capture_faces.py), comparison, and distance calculations. Depends on dlib.
- `torch`: The core deep learning framework.
- `facenet-pytorch`: Provides pre-trained MTCNN (detection) and InceptionResnetV1 (embedding) models.
- `numpy`: For numerical operations on image and embedding arrays.
- `Pillow` (PIL): Used for image manipulation and compatibility between OpenCV and PyTorch.
- `imutils`: For convenience functions like resizing video frames.
