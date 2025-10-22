# RBC26_Recruitment_Face_Recognition_Project

## Real-Time Face Recognition Project

This project implements a complete real-time face recognition pipeline using Python, OpenCV, and the FaceNet deep learning model. The system is broken into three main scripts:

1. *capture_faces.py*: Collects and stores face images to build a dataset.
2. *encode_faces.py*: Processes the collected images to create known face embeddings.
3. *recognize_faces.py*: Performs real-time face recognition using a live webcam feed.

## How It Works

The project uses a modern deep learning approach for high-accuracy face recognition:

- Face Detection: It uses a Multi-Task Cascaded Neural Network (MTCNN) from the facenet-pytorch library to accurately detect and crop faces from images and video frames.
- Face Embedding: For each detected face, it uses a pre-trained InceptionResnetV1 (FaceNet) model to compute a 512-dimension vector (an "embedding") that numerically represents the face.
- Face Matching: The recognition script compares the embeddings of faces seen on the webcam against a pre-computed database of "known" embeddings. A distance calculation (with a tolerance) determines if the face is a match or "Unknown".

## Installation

1. Clone the Repository
   ```bash
   git clone https://github.com/your-username/RBC26_Recruitment_Face_Recognition_Project.git
   cd RBC26_Recruitment_Face_Recognition_Project
   ```
2. Create a Virtual Environment (Recommended)
   ```bash
   python 3.11 -m venv face_rec
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   face_rec\Scripts\activate
   ```
3. Install the pre-compiled package for *dlib*
   ```bash
   python -m pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
   ```
4. Install Dependencies All required libraries are listed in *requirements.txt*. Install them using *pip*:
   ```bash
   pip install -r requirements.txt
   ```
   (If you are using a CUDA-enabled NVIDIA GPU, torch should automatically install with GPU support, which will make the process much faster.)

## Usage (Step-by-Step)

Follow these steps in order to run the project.

### Step 1: Capture Faces

First, you need to build a dataset of known faces. Run the *capture_faces.py* script.

```bash
python capture_faces.py
```

- You will be prompted to enter a name for the person.
- A new folder with this name will be created in the *dataset/ directory*.
- A webcam window will open. Position your face in the frame.
- Press the s key to save a frame. The script will only save when exactly one face is detected.
- Collect around 50 images for good results.
- Press q to quit when finished.

Repeat this process for each person you want to add to the dataset.

### Step 2: Encode Faces

Next, process the images you just saved into a single *"encodings"* file that the recognizer can use.

```bash
python encode_faces.py
```

- This script will scan the *dataset/ directory*.
- For each image, it will detect the face, compute its 512-d embedding, and store it.
- All embeddings and their corresponding names are saved to *encodings_facenet.pickle*.
- You only need to run this script once after you've added or changed images in the dataset folder.

### Step 3: Recognize Faces

Finally, run the real-time recognition script.

```bash
python recognize_faces.py
```

- This will load the *encodings_facenet.pickle file*.
- It will open your webcam and begin detecting all faces in the frame.
- For each face, it will compute its embedding and compare it to all known embeddings.
- A green box (for known faces) or red box (for "Unknown" faces) will be drawn, along with a name label.
- Press q to quit the video stream.

### Core Dependencies

- *opencv-python*: For capturing and displaying video from the webcam.
- *face-recognition*: Used for its efficient face comparison and distance calculation functions.
- *torch*: The core deep learning framework.
- *facenet-pytorch*: A package providing pre-trained MTCNN (for detection) and InceptionResnetV1 (for embedding) models.
- *numpy*: For numerical operations on image and embedding arrays.
- *Pillow* (PIL): Used for image manipulation and compatibility between OpenCV and PyTorch.
- *imutils*: For convenience functions like resizing video frames.
