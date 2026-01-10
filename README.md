# 🔐 Access-Control-System — Face Recognition Door Access

A lightweight, real-time face recognition access control system using OpenCV with YuNet for face detection and SFace for face embeddings.

This repository provides scripts to register users (capture faces + compute embeddings) and to run a live access control demo that recognizes the closest face to the camera and simulates door opening/closing.

Badges: Python 3.8+ · OpenCV 4.8+ · MIT License

## Table of contents

- About
- Features
- Quick start
- Usage
  - Register a user
  - Run access control (detection)
- Configuration
- Project structure
- How it works (brief)
- Troubleshooting & tips
- Contributing
- License

## About

This project demonstrates a complete pipeline for camera-based face access control:
- Detect faces using YuNet (with landmarks and confidence)
- Extract 128-D embeddings with SFace
- Select the closest face (largest bounding box) for recognition
- Compare embeddings with a simple cosine similarity threshold to grant or deny access

It’s designed to be simple to run on CPU and easy to extend for integration with real hardware (e.g., smart lock firmware in `smart_lock_firmware/`).

## Features

- Real-time face detection with landmark visualization (YuNet)
- Robust face embeddings via SFace (128-D)
- Closest-face prioritization (largest bounding box)
- Multi-user registration and per-user averaged embeddings
- Adjustable similarity threshold at runtime
- Auto-download of required models if missing
- Simple serialized embeddings database (`embeddings/face_database.pkl`)

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Verify OpenCV:

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

3. Register at least one user (see next section) then run the detector:

```bash
# register a user
python faceRegistration.py

# run access control
python faceDetection.py
```

## Usage

### Register a user

1. Open `config.yml` and set the user name and target images, e.g.:

```yaml
user:
  name: "alice"
  target_images: 20
```

2. Run registration:

```bash
python faceRegistration.py
```

Controls and notes:
- Position the person in front of the camera.
- Press `s` to start capture (the script will save several face crops and compute embeddings).
- Press `q` to quit at any time.
- The script computes per-capture embeddings and stores an average embedding in `embeddings/`.

### Run access control (real-time detection)

```bash
python faceDetection.py
```

While running:
- The system shows bounding boxes, landmarks and similarity scores.
- The closest face (largest BB) is used for recognition.
- Use `+` / `-` keys to raise/lower the similarity threshold.
- `ACCESS GRANTED` (green) appears when similarity >= threshold.

## Configuration

- `config.yml` controls user name, number of frames captured for registration, model paths, and threshold settings. Edit it before running registration or detection if you need custom behavior.
- Models are stored under `models/` and will be downloaded automatically if missing.

## Project structure

Top-level layout (key files):

```
Access-Control-System/
├─ faceRegistration.py      # Capture faces and register a user
├─ faceDetection.py         # Run live detection + recognition
├─ utils.py                 # Helper functions (I/O, drawing, embeddings)
├─ config.yml               # Runtime config (user name, thresholds)
├─ requirements.txt         # Python dependencies
├─ models/                  # ONNX models (YuNet, SFace)
├─ registered_faces/        # Saved face crops per user
├─ embeddings/              # face_database.pkl (serialized embeddings)
└─ smart_lock_firmware/     # Example Arduino/firmware files
```

## How it works (brief)

1. YuNet detects faces and 5-point landmarks in each frame.
2. The face with the largest bounding box (closest) is selected.
3. The face crop is fed to SFace to produce a 128-D embedding.
4. Cosine similarity compares the query embedding to stored per-user embeddings.
5. If similarity >= configurable threshold, access is granted.

Cosine similarity:

\\[ similarity = (A · B) / (||A|| ||B||) \\\]

Higher similarity indicates more likely same person. Default threshold is conservative but can be tuned.

## Troubleshooting & tips

- If no faces are detected: check camera access and lighting. Try increasing camera exposure or move closer.
- If recognition is inconsistent: capture more diverse images during registration (angles, lighting, expressions).
- Ensure `opencv-contrib-python` is installed (required for some face modules):

```bash
pip install opencv-contrib-python
```

- If models fail to download automatically, manually place the required ONNX files into `models/`:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`

## Contributing

Contributions welcome. Suggested improvements:

- Add a small web UI for remote monitoring
- Add encrypted on-disk storage for embeddings
- Integrate with a physical relay to trigger a real lock

Please open issues or pull requests with clear descriptions and tests where possible.

## License

MIT — see `LICENSE` (add one if not present).

---

If you'd like, I can also:
- add a short quickstart script or helper (Makefile) to simplify running the steps,
- or generate a minimal `LICENSE` file and small example showing how the embeddings are stored.

