"""
Face Detection and Recognition System - Main detection and verification.
"""

import cv2
import argparse
import numpy as np
from datetime import datetime
from utils import (
    load_config,
    print_config_summary,
    load_database,
    get_database_path,
    ensure_models_downloaded,
    load_models,
    setup_camera,
    get_closest_face,
    process_face_for_recognition,
    verify_face,
    draw_all_faces,
    draw_face_info,
    draw_similarity_panel,
    draw_main_ui
)
from MQTTPublisher import MQTTPublisher

class FaceDetectionSystem:
    """
    Face Detection and Recognition System class.
    
    Handles face detection, recognition, and access control with real-time video processing.
    """
    
    def __init__(self, threshold=None, camera_index=None, config_path='config.yml'):
        """
        Initialize the Face Detection System.
        
        Args:
            threshold: Override similarity threshold from config
            camera_index: Override camera index from config
            config_path: Path to configuration file
        """
        self.door_open = False
        self.last_detected_user = None
        
        # Load configuration using utility function
        self.config = load_config(config_path)
        
        # Override config if arguments provided
        if camera_index is not None:
            self.config['camera']['index'] = camera_index
        
        self.current_threshold = threshold if threshold is not None else self.config['detection']['similarity_threshold']
        
        # Print configuration using utility function
        print_config_summary(self.config, self.current_threshold)
        
        # Download and load models using utility functions
        yunet_path, sface_path = ensure_models_downloaded(self.config)
        self.detector, self.recognizer = load_models(
            yunet_path,
            sface_path if self.config['recognition']['enabled'] else None,
            tuple(self.config['yunet']['input_size']),
            self.config['yunet']['score_threshold'],
            self.config['yunet']['nms_threshold'],
            self.config['yunet']['top_k']
        )
        
        if self.recognizer is None:
            print("❌ Cannot run detection without recognition model")
            exit(1)
        
        # Load database using utility functions
        database_path = get_database_path(self.config)
        self.database = load_database(database_path)
        
        if not self.database:
            print("❌ Error: Face database is empty!")
            print("👉 Please run: python faceRegistration.py --register 'Your Name'")
            exit(1)
        
        print(f"👥 Registered users: {list(self.database.keys())}")
        
        # Setup camera using utility function
        self.cap = setup_camera(
            self.config['camera']['index'],
            self.config['camera']['width'],
            self.config['camera']['height'],
            self.config['camera']['fps']
        )
        
        # Update detector input size
        ret, test_frame = self.cap.read()
        if ret:
            h, w = test_frame.shape[:2]
            self.detector.setInputSize((w, h))

        # Setup MQTT publisher if enabled
        self.mqtt_publisher = None
        if self.config.get('mqtt', {}).get('enabled', False):
            mqtt_config = self.config['mqtt']
            cafile_path = "/etc/mosquitto/certs/mosquitto.crt"  # Path to CA cert
            self.mqtt_publisher = MQTTPublisher(
                broker_address=mqtt_config['broker'],
                port=mqtt_config['port'],
                topic=mqtt_config['topic'],
                client_id=mqtt_config['client_id'],
                username="espuser",          # your Mosquitto user
                password="esppass",
                cafile=cafile_path
            )
            if not self.mqtt_publisher.connect():
                print("⚠️  Running without MQTT (connection failed11147)")
                self.mqtt_publisher = None

    
    def _process_frame(self, frame):
        """
        Process a single frame for face detection and recognition.
        """
        h, w = frame.shape[:2]
        
        # Detect faces
        _, faces = self.detector.detect(frame)
        num_faces = len(faces) if faces is not None else 0
        if num_faces == 0:
            if self.last_detected_user != "NO_ONE":
                # print("ℹ️  Status: No Face Detected (Idle)") # Optional debug print
                self.last_detected_user = "NO_ONE"
                self.door_open = False
                
                if self.mqtt_publisher:
                    # Send "NO_ONE" decision
                    self.mqtt_publisher.publish_decision(None, "NO_ONE", 0.0)
        # Find closest face
        closest_face = get_closest_face(faces)
        
        # Draw all non-closest faces in gray
        draw_all_faces(frame, faces, closest_face)
        
        # Process closest face
        result_text = "NO FACE DETECTED"
        result_color = (128, 128, 128)
        access_status = "STANDBY"
        best_similarity = 0.0
        all_scores = {}
        
        if closest_face is not None:
            # --- EXISTING LOGIC FOR DETECTED FACES ---
            alignment_size = tuple(self.config['detection']['face_alignment_size'])
            embedding = process_face_for_recognition(frame, closest_face, self.recognizer, alignment_size)
            
            # Verify against database
            is_match, best_match, best_similarity, all_scores = verify_face(
                embedding, self.database, self.current_threshold
            )
            
            if is_match:
                result_color = (0, 255, 0)  # Green
                result_text = f"MATCH: {best_match}"
                access_status = "✓ ACCESS GRANTED"
                self.door_open = True
                
                # Only log/publish if the user changed or status changed
                if self.last_detected_user != best_match:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"✅ {timestamp} - ACCESS GRANTED: {best_match} (Similarity: {best_similarity:.3f})")
                    self.last_detected_user = best_match
                    
                    if self.mqtt_publisher:
                        self.mqtt_publisher.publish_decision(best_match, "GRANTED", best_similarity)
            else:
                result_color = (0, 0, 255)  # Red
                result_text = "NO MATCH"
                access_status = "✗ ACCESS DENIED"
                self.door_open = False
                
                if best_match:
                    result_text += f" (Best: {best_match})"
                
                # Only log/publish if we weren't already in UNKNOWN state
                if self.last_detected_user != "UNKNOWN":
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"❌ {timestamp} - ACCESS DENIED: Unknown (Best: {best_match})")
                    self.last_detected_user = "UNKNOWN"

                    if self.mqtt_publisher:
                        self.mqtt_publisher.publish_decision(best_match, "DENIED", best_similarity)
            
            # Draw face info
            draw_face_info(frame, closest_face, result_text, best_similarity, 
                          access_status, result_color)
            
            # Draw similarity panel
            draw_similarity_panel(frame, all_scores, self.current_threshold)
            
        #else:
            # --- NEW LOGIC: NO FACE DETECTED ---
            # If we previously detected someone, send "NO_ONE" signal to turn LED Blue
        #    if self.last_detected_user != "NO_ONE":
                # print("ℹ️  Status: No Face Detected (Idle)") # Optional debug print
        #        self.last_detected_user = "NO_ONE"
        #        self.door_open = False
                
        #        if self.mqtt_publisher:
                    # Send "NO_ONE" decision
        #            self.mqtt_publisher.publish_decision(None, "NO_ONE", 0.0)

        # Draw main UI
        draw_main_ui(frame, self.door_open, num_faces, self.current_threshold,
                    self.config['yunet']['score_threshold'], len(self.database))
        
        return frame, num_faces
    
    def _handle_key_input(self, key):
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            bool: True if system should continue, False if should quit
        """
        if key == ord('q'):
            print("👋 System shutting down...")
            return False
        
        # Adjust threshold
        elif key == ord('+') or key == ord('='):
            step = self.config['detection']['threshold_adjustment_step']
            self.current_threshold = min(1.0, self.current_threshold + step)
            print(f"⚙️  Threshold increased to: {self.current_threshold:.2f}")
        
        elif key == ord('-') or key == ord('_'):
            step = self.config['detection']['threshold_adjustment_step']
            self.current_threshold = max(0.0, self.current_threshold - step)
            print(f"⚙️  Threshold decreased to: {self.current_threshold:.2f}")
        
        return True
    
    def run(self):
        """Run the face detection and recognition system."""
        print("=" * 60)
        print("✅ Face Detection System Started")
        print("👉 Press 'q' to quit")
        print("👉 Press '+' or '-' to adjust similarity threshold")
        print("=" * 60)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera error")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame, num_faces = self._process_frame(frame)
                
                # Display result
                cv2.imshow("Face Detection System", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key_input(key):
                    break
        # ... existing code ...
                cv2.imshow("Face Detection System", processed_frame)
                
                # 👉 ADD THIS LINE to slow down sending speed
                import time
                time.sleep(0.1)  # Wait 100ms (Max 10 checks per second)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
        finally:
            self.cleanup()
        
        self._print_summary()
    
    def cleanup(self):
        """Release resources and cleanup."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def _print_summary(self):
        """Print system shutdown summary."""
        print("\n✅ System stopped successfully")
        print("=" * 60)
        print(f"Final threshold: {self.current_threshold:.2f}")
        if self.current_threshold != self.config['detection']['similarity_threshold']:
            print(f"💡 Consider updating config.yml threshold from "
                  f"{self.config['detection']['similarity_threshold']:.2f} to {self.current_threshold:.2f}")
        print("=" * 60)


def run_detection(threshold=None, camera_index=None, config_path='config.yml'):
    """
    Run face detection and recognition system (wrapper function for backward compatibility).
    
    Args:
        threshold: Override similarity threshold from config
        camera_index: Override camera index from config
        config_path: Path to configuration file
    """
    system = FaceDetectionSystem(threshold=threshold, camera_index=camera_index, config_path=config_path)
    system.run()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Face Detection and Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run with default settings:
    python faceDetection.py
    
  Use custom similarity threshold:
    python faceDetection.py --threshold 0.5
    
  Use different camera:
    python faceDetection.py --camera 1
    
  Use custom config file:
    python faceDetection.py --config my_config.yml
    
  Combine options:
    python faceDetection.py --threshold 0.45 --camera 0
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        metavar='PATH',
        help='Path to configuration file (default: config.yml)'
    )
    
    args = parser.parse_args()
    
    
    
    run_detection(config_path=args.config)


if __name__ == "__main__":
    main()