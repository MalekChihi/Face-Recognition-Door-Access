"""
Utility functions for face detection and recognition system.
"""

import cv2
import os
import yaml
import pickle
import urllib.request
import numpy as np
from typing import Optional, Tuple, Dict, List


# ============================== 
# CONFIGURATION MANAGEMENT
# ==============================

def load_config(config_path='config.yml') -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        SystemExit: If config file not found or invalid
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
        return config
    except FileNotFoundError:
        print(f"❌ Error: Configuration file '{config_path}' not found!")
        exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing configuration file: {e}")
        exit(1)


def print_config_summary(config: dict, threshold: float = 0.0):
    """
    Print configuration summary.
    
    Args:
        config: Configuration dictionary
        threshold: Optional threshold override
    """
    display_threshold = threshold if threshold is not None else config['detection']['similarity_threshold']
    
    print("=" * 60)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"🎥 Camera: Index {config['camera']['index']} "
          f"({config['camera']['width']}x{config['camera']['height']})")
    print(f"🔍 YuNet threshold: {config['yunet']['score_threshold']}")
    print(f"🧠 Recognition model: {config['recognition']['model_name']}")
    print(f"📊 Similarity threshold: {display_threshold}")
    print("=" * 60)


# ============================== 
# DATABASE MANAGEMENT
# ==============================

def load_database(database_path: str) -> dict:
    """
    Load face database from file.
    
    Args:
        database_path: Path to database pickle file
        
    Returns:
        Dictionary of {name: embedding}
    """
    if os.path.exists(database_path):
        with open(database_path, 'rb') as f:
            database = pickle.load(f)
        print(f"✅ Loaded existing database with {len(database)} users")
        return database
    else:
        print("📝 Creating new face database")
        return {}


def save_database(database: dict, database_path: str):
    """
    Save face database to file.
    
    Args:
        database: Dictionary of {name: embedding}
        database_path: Path to save database
    """
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    with open(database_path, 'wb') as f:
        pickle.dump(database, f)
    print(f"✅ Database saved with {len(database)} users")


def get_database_path(config: dict) -> str:
    """
    Get database file path from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Full path to database file
    """
    return os.path.join(config['paths']['embeddings'], "face_database.pkl")


# ============================== 
# MODEL MANAGEMENT
# ==============================

def download_model(url: str, output_path: str) -> bool:
    """
    Download model file from URL.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📥 Downloading model from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Model downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def ensure_models_downloaded(config: dict) -> Tuple[str, Optional[str]]:
    """
    Ensure all required models are downloaded.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (yunet_path, sface_path)
        
    Raises:
        SystemExit: If YuNet download fails
    """
    models_path = config['paths']['models']
    os.makedirs(models_path, exist_ok=True)
    
    # YuNet model
    yunet_name = config['yunet']['model_name']
    yunet_path = os.path.join(models_path, yunet_name)
    if not os.path.exists(yunet_path):
        url = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{yunet_name}"
        if not download_model(url, yunet_path):
            print("❌ Failed to download YuNet model")
            exit(1)
    else:
        print(f"✅ YuNet model found: {yunet_name}")
    
    # SFace model
    sface_name = config['recognition']['model_name']
    sface_path = os.path.join(models_path, sface_name)
    if config['recognition']['enabled'] and not os.path.exists(sface_path):
        url = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/{sface_name}"
        if not download_model(url, sface_path):
            print("⚠️  Recognition model not available")
            return yunet_path, None
    else:
        print(f"✅ Recognition model found: {sface_name}")
    
    return yunet_path, sface_path


def load_models(yunet_path: str, sface_path: Optional[str], 
                input_size: Tuple[int, int], score_threshold: float,
                nms_threshold: float, top_k: int) -> Tuple[cv2.FaceDetectorYN, Optional[cv2.FaceRecognizerSF]]:
    """
    Load YuNet face detector and SFace recognizer.
    
    Args:
        yunet_path: Path to YuNet model
        sface_path: Path to SFace model (None to skip)
        input_size: Input size for detector
        score_threshold: Detection score threshold
        nms_threshold: NMS threshold
        top_k: Maximum faces to detect
        
    Returns:
        Tuple of (detector, recognizer)
    """
    print("🔄 Loading YuNet face detector...")
    detector = cv2.FaceDetectorYN.create(
        yunet_path,
        "",
        input_size,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k
    )
    print("✅ YuNet face detector loaded")
    
    recognizer = None
    if sface_path:
        try:
            recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
            print("✅ SFace recognizer loaded")
        except Exception as e:
            print(f"⚠️  Could not load SFace recognizer: {e}")
    
    return detector, recognizer


# ============================== 
# CAMERA MANAGEMENT
# ==============================

def setup_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """
    Initialize and configure camera.
    
    Args:
        camera_index: Camera device index
        width: Frame width
        height: Frame height
        fps: Frames per second
        
    Returns:
        Configured VideoCapture object
        
    Raises:
        RuntimeError: If camera cannot be opened
    """
    print(f"🔄 Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    
    ret, test_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read from camera")
    
    print("✅ Camera initialized successfully")
    return cap


# ============================== 
# FACE DETECTION AND PROCESSING
# ==============================

def get_closest_face(faces: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Get the face closest to the camera (largest face area).
    
    Args:
        faces: Array of detected faces from YuNet detector
        
    Returns:
        The face with largest area, or None if no faces detected
    """
    if faces is None or len(faces) == 0:
        return None
    
    largest_face = None
    max_area = 0
    
    for face in faces:
        w, h = face[2], face[3]
        area = w * h
        if area > max_area:
            max_area = area
            largest_face = face
    
    return largest_face


def extract_face_roi(frame: np.ndarray, face: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract face region of interest from frame.
    
    Args:
        frame: Input frame
        face: Detected face data from YuNet
        
    Returns:
        Tuple of (face_roi, (x, y, w, h))
    """
    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
    
    # Ensure coordinates are within frame bounds
    frame_h, frame_w = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    
    face_roi = frame[y:y+h, x:x+w]
    return face_roi, (x, y, w, h)


def extract_embedding(face_roi: np.ndarray, aligned_face: np.ndarray, 
                      recognizer: Optional[cv2.FaceRecognizerSF]) -> np.ndarray:
    """
    Extract face embedding using SFace recognizer.
    
    Args:
        face_roi: Original face region
        aligned_face: Aligned face (112x112)
        recognizer: SFace recognizer model
        
    Returns:
        Face embedding vector
    """
    if recognizer is None:
        # Fallback to basic histogram if no recognizer
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        return resized.flatten().astype(np.float32)
    
    return recognizer.feature(aligned_face)


def process_face_for_recognition(frame: np.ndarray, face: np.ndarray, 
                                  recognizer: Optional[cv2.FaceRecognizerSF],
                                  alignment_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Complete pipeline: extract face ROI, align, extract embedding, and normalize.
    
    Args:
        frame: Input frame
        face: Detected face data from YuNet
        recognizer: SFace recognizer model
        alignment_size: Size for face alignment (default: (112, 112))
        
    Returns:
        Normalized face embedding
    """
    # Extract face ROI
    face_roi, _ = extract_face_roi(frame, face)
    
    # Align face
    aligned_face = cv2.resize(face_roi, alignment_size)
    
    # Extract embedding
    embedding = extract_embedding(face_roi, aligned_face, recognizer)
    
    # Normalize
    return normalize_embedding(embedding)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length.
    
    Args:
        embedding: Input embedding vector
        
    Returns:
        Normalized embedding (flattened to 1D)
    """
    # Ensure embedding is 1D
    embedding = embedding.flatten()
    
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Cosine similarity score (0-1)
    """
    # Ensure embeddings are 1D
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(np.clip(similarity, 0, 1))


def verify_face(embedding: np.ndarray, database: Dict[str, np.ndarray], 
                threshold: float) -> Tuple[bool, Optional[str], float, Dict[str, float]]:
    """
    Verify face against database.
    
    Args:
        embedding: Face embedding to verify
        database: Dictionary of {name: embedding}
        threshold: Similarity threshold for matching
        
    Returns:
        Tuple of (is_match, best_match_name, best_similarity, all_scores)
    """
    if not database:
        return False, None, 0.0, {}
    
    all_scores = {}
    best_match = None
    best_similarity = 0.0
    
    for name, stored_embedding in database.items():
        similarity = compute_cosine_similarity(embedding, stored_embedding)
        all_scores[name] = similarity
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name
    
    is_match = best_similarity >= threshold
    return is_match, best_match, best_similarity, all_scores


# ============================== 
# VISUALIZATION AND UI
# ==============================

def draw_face_landmarks(frame: np.ndarray, face: np.ndarray, 
                        color: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """
    Draw face bounding box and landmarks.
    
    Args:
        frame: Frame to draw on
        face: Face data from YuNet (includes landmarks)
        color: Color for drawing
        
    Returns:
        Tuple of (x, y, w, h) of bounding box
    """
    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw landmarks (right eye, left eye, nose, right mouth, left mouth)
    landmarks = [
        (int(face[4]), int(face[5])),   # right eye
        (int(face[6]), int(face[7])),   # left eye
        (int(face[8]), int(face[9])),   # nose
        (int(face[10]), int(face[11])), # right mouth
        (int(face[12]), int(face[13]))  # left mouth
    ]
    
    for landmark in landmarks:
        cv2.circle(frame, landmark, 2, color, -1)
    
    return x, y, w, h


def draw_all_faces(frame: np.ndarray, faces: Optional[np.ndarray], 
                   closest_face: Optional[np.ndarray]):
    """
    Draw all non-closest faces in gray.
    
    Args:
        frame: Frame to draw on
        faces: All detected faces
        closest_face: The closest face (will not be drawn gray)
    """
    if faces is not None:
        for face in faces:
            is_closest = closest_face is not None and np.array_equal(face, closest_face)
            if not is_closest:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 1)


def draw_face_info(frame: np.ndarray, face: np.ndarray, result_text: str,
                   best_similarity: float, access_status: str, result_color: Tuple[int, int, int]):
    """
    Draw face information label.
    
    Args:
        frame: Frame to draw on
        face: Face data
        result_text: Recognition result text
        best_similarity: Best match similarity score
        access_status: Access status text
        result_color: Color for drawing
    """
    confidence = face[14]
    x, y, w, h = draw_face_landmarks(frame, face, result_color)
    
    # Label background
    label_height = 90
    cv2.rectangle(frame, (x, y - label_height), (x + w, y), result_color, -1)
    
    # Text
    cv2.putText(frame, result_text, (x + 5, y - 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Similarity: {best_similarity:.3f}", (x + 5, y - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (x + 5, y - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, access_status, (x + 5, y - 2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_similarity_panel(frame: np.ndarray, all_scores: Dict[str, float], threshold: float):
    """
    Draw similarity scores panel.
    
    Args:
        frame: Frame to draw on
        all_scores: Dictionary of {name: score}
        threshold: Current threshold
    """
    h, w = frame.shape[:2]
    panel_x = w - 280
    panel_y = 150
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 10, panel_y - 30), 
                 (w - 10, panel_y + len(all_scores) * 25 + 10), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "SIMILARITY SCORES:", (panel_x, panel_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    panel_y += 25
    for user, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        color = (0, 255, 0) if score >= threshold else (0, 0, 255)
        text = f"{user}: {score:.3f}"
        cv2.putText(frame, text, (panel_x, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        panel_y += 20


def draw_main_ui(frame: np.ndarray, door_open: bool, num_faces: int, 
                 threshold: float, yunet_score: float, num_users: int):
    """
    Draw main UI overlay.
    
    Args:
        frame: Frame to draw on
        door_open: Whether door is open
        num_faces: Number of detected faces
        threshold: Current similarity threshold
        yunet_score: YuNet score threshold
        num_users: Number of registered users
    """
    h, w = frame.shape[:2]
    
    # Door status
    door_status = "🚪 DOOR: OPEN" if door_open else "🚪 DOOR: CLOSED"
    door_color = (0, 255, 0) if door_open else (0, 0, 255)
    
    # Top overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, door_status, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, door_color, 2)
    cv2.putText(frame, f"Faces: {num_faces} | Threshold: {threshold:.2f}", 
               (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"YuNet (score: {yunet_score}) + SFace", 
               (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"Users: {num_users}", 
               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_registration_ui(frame: np.ndarray, name: str, captured: int, 
                         total: int, closest_face: Optional[np.ndarray]):
    """
    Draw registration UI overlay.
    
    Args:
        frame: Frame to draw on
        name: Name of person being registered
        captured: Number of samples captured
        total: Total samples needed
        closest_face: Current closest face
    """
    h, w = frame.shape[:2]
    
    # Top overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, f"Registering: {name}", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Captured: {captured}/{total}", (20, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw face with instructions
    if closest_face is not None:
        confidence = closest_face[14]
        color = (0, 255, 0) if confidence > 0.9 else (0, 165, 255)
        x, y, w_box, h_box = draw_face_landmarks(frame, closest_face, color)
        cv2.putText(frame, "Press SPACE to capture", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(frame, "NO FACE DETECTED", (w//2 - 150, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def show_capture_flash(frame: np.ndarray):
    """
    Show visual feedback when capturing a face.
    
    Args:
        frame: Frame to flash
    """
    h, w = frame.shape[:2]
    flash = frame.copy()
    cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), -1)
    cv2.addWeighted(flash, 0.5, frame, 0.5, 0, frame)
    cv2.imshow("Face Registration", frame)
    cv2.waitKey(100)