# 🔐 Face Recognition Access Control System

A real-time face recognition access control system using **YuNet** face detection and **SFace** face recognition with OpenCV. The system detects the closest face to the camera, extracts face embeddings (tensor representations), and compares them with registered faces to grant or deny access.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Advanced Features](#-advanced-features)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- ✅ **State-of-the-art Face Detection**: YuNet detector with facial landmarks
- ✅ **Accurate Face Recognition**: SFace model with 128-D embeddings
- ✅ **Closest Face Priority**: Always processes the face nearest to camera
- ✅ **Real-time Processing**: Fast detection and recognition
- ✅ **Multi-user Support**: Register and recognize multiple users
- ✅ **Access Control Simulation**: Door open/close based on recognition
- ✅ **Confidence Scoring**: Shows detection and recognition confidence
- ✅ **Visual Feedback**: Color-coded bounding boxes and facial landmarks
- ✅ **Access Logging**: Timestamp-based access attempt logging
- ✅ **Adjustable Threshold**: Runtime threshold adjustment
- ✅ **Auto Model Download**: Automatically downloads required models

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CAMERA INPUT                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              YUNET FACE DETECTION                            │
│   • Detects all faces in frame                              │
│   • Extracts facial landmarks (5 points)                    │
│   • Returns confidence scores                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           SELECT CLOSEST FACE                                │
│   • Based on bounding box area (largest = closest)          │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴────────────────┐
        │                                │
        ▼                                ▼
┌──────────────────┐          ┌──────────────────────┐
│  REGISTRATION    │          │   DETECTION          │
│  • Save images   │          │   • Extract ROI      │
│  • Extract       │          │   • Extract          │
│    embeddings    │          │     embedding        │
│  • Store in DB   │          │   • Compare with DB  │
└──────────────────┘          └──────────┬───────────┘
                                         │
                                         ▼
                        ┌────────────────────────────┐
                        │  COSINE SIMILARITY         │
                        │  • Compare embeddings      │
                        │  • Calculate scores        │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────┐
                        │  ACCESS DECISION           │
                        │  POSITIVE: Match found     │
                        │  NEGATIVE: No match        │
                        └────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- Internet connection (for first-time model download)

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd face-recognition-access-control

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `opencv-contrib-python>=4.8.0` - Computer vision library with extra modules
- `numpy>=1.24.0` - Numerical computing
- `Pillow>=10.0.0` - Image processing
- `PyYAML>=6.0` - YAML configuration file parsing

**Note:** Make sure to install `opencv-contrib-python` (not just `opencv-python`) to get the face recognition modules.

### Step 3: Verify Installation

```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

Expected output: `OpenCV version: 4.8.x` or higher

## 📁 Project Structure

```
face-recognition-access-control/
│
├── faceRegistration.py          # Register new users
├── faceDetection.py             # Run access control system
├── utils.py                     # Helper functions
├── config.yml                   # Configuration file ⭐ NEW
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── models/                      # AI models (auto-downloaded)
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognizer_sface_2021dec.onnx
│
├── registered_faces/            # Face images (auto-created)
│   ├── ahmed/
│   │   ├── face_0_*.jpg
│   │   ├── face_1_*.jpg
│   │   └── ...
│   ├── john/
│   └── ...
│
└── embeddings/                  # Face database (auto-created)
    └── face_database.pkl        # Serialized embeddings
```

## 🎯 Usage

### 1. Register a New User

Edit `config.yml` and change the user name:

```yaml
user:
  name: "ahmed"  # Change to the person's name
  target_images: 20
```

Run the registration script:

```bash
python faceRegistration.py
```

**Instructions:**
1. Position your face in front of the camera
2. Wait for face detection (green bounding box with landmarks)
3. Press **'s'** to start capturing 20 images
4. The system will automatically extract embeddings
5. Press **'q'** to quit anytime

**What happens:**
- Detects the **closest face** (largest bounding box)
- Captures 20 diverse images (with frame skipping)
- Extracts 128-D embedding for each image
- Computes average embedding for robust matching
- Saves images and embeddings to database

### 2. Run Face Detection

```bash
python faceDetection.py
```

**Instructions:**
- System starts automatically detecting faces
- Green box + "ACCESS GRANTED" = Recognized user
- Red box + "ACCESS DENIED" = Unknown person
- Press **'+'** or **'-'** to adjust similarity threshold
- Press **'q'** to quit

**Real-time Display:**
- Door status (OPEN/CLOSED)
- Face detection count
- Similarity scores for all registered users
- Facial landmarks
- Confidence levels

### 3. Register Multiple Users

Repeat step 1 for each person by updating `config.yml`:

```yaml
# First user
user:
  name: "ahmed"

# Run: python faceRegistration.py

# Then change to next user
user:
  name: "john"

# Run: python faceRegistration.py again
```

The database automatically handles multiple users.

## 🔬 How It Works

### Face Detection (YuNet)

**YuNet** is a lightweight, high-accuracy face detector that:
- Detects faces at various scales and angles
- Extracts 5 facial landmarks: eyes, nose, mouth corners
- Provides confidence scores for each detection
- Runs in real-time on CPU

**Detection Process:**
```
Frame → YuNet → Faces + Landmarks + Confidence
```

### Face Recognition (SFace)

**SFace** extracts 128-dimensional feature vectors (embeddings) that represent faces:
- Trained on millions of faces
- Robust to lighting, pose, expression variations
- Maps similar faces to nearby points in 128-D space
- Enables fast comparison via vector math

**Recognition Process:**
```
Face Image → SFace → 128-D Embedding → Normalization → Unit Vector
```

### Closest Face Selection

The system prioritizes the face with the **largest bounding box area**:

```python
area = width × height
closest_face = face_with_max_area
```

**Why area?** Larger bounding box typically means the face is closer to the camera.

### Face Comparison

**Cosine Similarity** measures the angle between two embedding vectors:

```
similarity = (A · B) / (||A|| × ||B||)

where:
  A = query embedding
  B = registered embedding
  · = dot product
  ||·|| = L2 norm (vector length)
```

**Interpretation:**
- `1.0` = Identical faces (0° angle)
- `0.0` = Completely different (90° angle)
- `-1.0` = Opposite (180° angle)

**Decision Logic:**
```python
if similarity >= threshold:  # Default: 0.4
    return "POSITIVE - ACCESS GRANTED"
else:
    return "NEGATIVE - ACCESS DENIED"
```

### Database Structure

Face embeddings are stored in a pickle file:

```python
face_database = {
    'ahmed': {
        'embedding': np.array([...]),          # Average 128-D vector
        'embeddings_list': [array1, array2, ...],  # All 20 embeddings
        'num_samples': 20,
        'timestamp': '2024-12-06T14:30:25',
        'method': 'YuNet + SFace'
    },
    'john': {
        ...
    }
}
```

## 🎓 Learn More

- [OpenCV Documentation](https://docs.opencv.org/)
- [YuNet Paper](https://github.com/ShiqiYu/libfacedetection)
- [Face Recognition Theory](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)

---
