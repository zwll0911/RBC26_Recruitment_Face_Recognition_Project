# RBC26_Recruitment_Face_Recognition_Project

## Real-Time Face Recognition Project

This project implements a complete real-time face recognition pipeline using Python, OpenCV, and the FaceNet deep learning model. The system is broken into three main scripts:

1. capture_faces.py: Collects and stores face images to build a dataset.
2. encode_faces.py: Processes the collected images to create known face embeddings.
3. recognize_faces.py: Performs real-time face recognition using a live webcam feed.

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
