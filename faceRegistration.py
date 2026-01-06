"""
Face registration system - Register new faces to the database.
"""

import cv2
import argparse
import numpy as np
from datetime import datetime
from typing import Optional
from utils import (
    load_config,
    load_database,
    save_database,
    get_database_path,
    ensure_models_downloaded,
    load_models,
    setup_camera,
    get_closest_face,
    extract_face_roi,
    extract_embedding,
    normalize_embedding,
    draw_registration_ui,
    show_capture_flash
)


class FaceRegistration:
    """
    Face registration system for capturing and storing face embeddings.
    
    Attributes:
        name: Name of person to register
        num_samples: Number of face samples to capture
        config: Configuration dictionary
        database_path: Path to database file
        detector: YuNet face detector
        recognizer: SFace face recognizer
        database: Face database dictionary
        embeddings: List of captured embeddings
        captured: Number of samples captured
    """
    
    def __init__(self, name: str, num_samples: int = 5, config_path: str = 'config.yml'):
        """
        Initialize face registration system.
        
        Args:
            name: Name of person to register
            num_samples: Number of face samples to capture (default: 5)
            config_path: Path to configuration file (default: 'config.yml')
        """
        self.name = name
        self.num_samples = num_samples
        self.config = load_config(config_path)
        self.database_path = get_database_path(self.config)
        
        # Models
        self.detector = None
        self.recognizer = None
        
        # Data
        self.database = {}
        self.embeddings = []
        self.captured = 0
        
        # Camera
        self.cap = None
    
    def setup(self):
        """
        Setup models, database, and camera.
        
        Raises:
            SystemExit: If recognition model not available
        """
        # Ensure models are downloaded
        yunet_path, sface_path = ensure_models_downloaded(self.config)
        
        # Load models
        self.detector, self.recognizer = load_models(
            yunet_path,
            sface_path,
            tuple(self.config['yunet']['input_size']),
            self.config['yunet']['score_threshold'],
            self.config['yunet']['nms_threshold'],
            self.config['yunet']['top_k']
        )
        
        if self.recognizer is None:
            print("❌ Cannot register without recognition model")
            exit(1)
        
        # Load database
        self.database = load_database(self.database_path)
        
        # Check if user already exists
        if self.name in self.database:
            response = input(f"⚠️  User '{self.name}' already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("❌ Registration cancelled")
                exit(0)
        
        # Setup camera
        self.cap = setup_camera(
            self.config['camera']['index'],
            self.config['camera']['width'],
            self.config['camera']['height'],
            self.config['camera']['fps']
        )
        
        # Update detector input size
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
    
    def print_instructions(self):
        """Print registration instructions."""
        print("\n" + "=" * 60)
        print(f"📸 FACE REGISTRATION: {self.name}")
        print("=" * 60)
        print(f"📋 Instructions:")
        print(f"   • Position your face in the center")
        print(f"   • Look directly at the camera")
        print(f"   • Press SPACE to capture ({self.num_samples} samples needed)")
        print(f"   • Press 'q' to quit")
        print("=" * 60)
    
    def capture_sample(self, frame: np.ndarray, closest_face: np.ndarray) -> bool:
        """
        Capture a single face sample.
        
        Args:
            frame: Current video frame
            closest_face: Detected face data
            
        Returns:
            True if sample captured successfully
        """
        # Extract and process face
        face_roi, _ = extract_face_roi(frame, closest_face)
        aligned_face = cv2.resize(face_roi, (112, 112))
        embedding = extract_embedding(face_roi, aligned_face, self.recognizer)
        embedding = normalize_embedding(embedding)
        
        # Store embedding
        self.embeddings.append(embedding)
        self.captured += 1
        
        print(f"✅ Captured sample {self.captured}/{self.num_samples}")
        
        # Visual feedback
        show_capture_flash(frame)
        
        return True
    
    def run(self) -> bool:
        """
        Run the registration process.
        
        Returns:
            True if registration completed successfully, False otherwise
        """
        self.print_instructions()
        
        if self.cap is None:
            print("❌ Camera not initialized")
            return False
        
        while self.captured < self.num_samples:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Camera error")
                return False
            
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            if self.detector is None:
                print("❌ Detector not initialized")
                return False
            _, faces = self.detector.detect(frame)
            
            # Get closest face
            closest_face = get_closest_face(faces)
            
            # Draw UI
            draw_registration_ui(frame, self.name, self.captured, 
                               self.num_samples, closest_face)
            
            cv2.imshow("Face Registration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n❌ Registration cancelled by user")
                return False
            
            elif key == ord(' ') and closest_face is not None:
                self.capture_sample(frame, closest_face)
        
        return True
    
    def save(self):
        """Save registered face to database."""
        if self.captured == self.num_samples:
            # Average embeddings
            avg_embedding = np.mean(self.embeddings, axis=0)
            avg_embedding = normalize_embedding(avg_embedding)
            
            # Save to database
            self.database[self.name] = avg_embedding
            save_database(self.database, self.database_path)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("\n" + "=" * 60)
            print(f"✅ SUCCESS: {self.name} registered successfully!")
            print(f"⏰ Timestamp: {timestamp}")
            print(f"👥 Total users in database: {len(self.database)}")
            print("=" * 60)
            return True
        else:
            print("\n❌ Registration incomplete")
            return False
    
    def cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def register(self) -> bool:
        """
        Complete registration workflow: setup, capture, save, cleanup.
        
        Returns:
            True if registration successful, False otherwise
        """
        try:
            self.setup()
            success = self.run()
            if success:
                self.save()
            return success
        finally:
            self.cleanup()


# ============================== 
# UTILITY FUNCTIONS
# ==============================

def list_users(config_path: str = 'config.yml'):
    """
    List all registered users.
    
    Args:
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    database_path = get_database_path(config)
    database = load_database(database_path)
    
    if not database:
        print("📝 Database is empty. No users registered.")
        return
    
    print("\n" + "=" * 60)
    print("👥 REGISTERED USERS")
    print("=" * 60)
    for i, name in enumerate(database.keys(), 1):
        print(f"{i}. {name}")
    print("=" * 60)
    print(f"Total: {len(database)} users")


def delete_user(name: str, config_path: str = 'config.yml'):
    """
    Delete a user from the database.
    
    Args:
        name: Name of user to delete
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    database_path = get_database_path(config)
    database = load_database(database_path)
    
    if name not in database:
        print(f"❌ User '{name}' not found in database")
        return
    
    response = input(f"⚠️  Delete user '{name}'? (y/n): ")
    if response.lower() == 'y':
        del database[name]
        save_database(database, database_path)
        print(f"✅ User '{name}' deleted successfully")
    else:
        print("❌ Deletion cancelled")


# ============================== 
# MAIN
# ==============================

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Face Registration System - Register faces to the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Register a new user (interactive):
    python faceRegistration.py --register
    
  Register with name provided:
    python faceRegistration.py --register "John Doe"
    
  Register with custom sample count:
    python faceRegistration.py --register "Jane Smith" --samples 10
    
  List all registered users:
    python faceRegistration.py --list
    
  Delete a user (interactive):
    python faceRegistration.py --delete
    
  Delete with name provided:
    python faceRegistration.py --delete "John Doe"
        """
    )
    
    parser.add_argument(
        '--register', '-r',
        nargs='?',  # Makes the name optional (0 or 1 argument)
        const='',   # If --register is provided without name, use empty string
        metavar='NAME',
        help='Register a new user (provide name or will prompt)'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=5,
        metavar='N',
        help='Number of face samples to capture (default: 5)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all registered users'
    )
    
    parser.add_argument(
        '--delete', '-d',
        nargs='?',  # Makes the name optional (0 or 1 argument)
        const='',   # If --delete is provided without name, use empty string
        metavar='NAME',
        help='Delete a user from the database (provide name or will prompt)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        metavar='PATH',
        help='Path to configuration file (default: config.yml)'
    )
    
    args = parser.parse_args()
    
    # Execute based on arguments
    if args.register is not None:  # Check if --register was used
        # If name not provided, prompt for it
        name = args.register if args.register else input("Enter name to register: ").strip()
        
        if not name:
            print("❌ Error: Name cannot be empty")
            exit(1)
        
        registrar = FaceRegistration(name, args.samples, args.config)
        registrar.register()
        
    elif args.list:
        list_users(args.config)
        
    elif args.delete is not None:  # Check if --delete was used
        # If name not provided, prompt for it
        name = args.delete if args.delete else input("Enter name to delete: ").strip()
        
        if not name:
            print("❌ Error: Name cannot be empty")
            exit(1)
        
        delete_user(name, args.config)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()