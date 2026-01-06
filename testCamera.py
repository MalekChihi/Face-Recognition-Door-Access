import cv2
import sys

print("=" * 60)
print("CAMERA & FACE DETECTION TEST")
print("=" * 60)

# Test 1: Check OpenCV version
print(f"✓ OpenCV version: {cv2.__version__}")

# Test 2: Load face cascade
print("\n📋 Testing face detector...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("❌ ERROR: Could not load Haar Cascade!")
    print("Try reinstalling: pip install --force-reinstall opencv-python")
    sys.exit(1)
else:
    print("✅ Face cascade loaded successfully")

# Test 3: Open camera
print("\n📷 Testing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot open camera 0")
    print("Trying camera 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open any camera")
        sys.exit(1)

print("✅ Camera opened successfully")

# Get camera properties
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"   Resolution: {int(width)}x{int(height)}")
print(f"   FPS: {fps}")

print("\n" + "=" * 60)
print("🎥 LIVE CAMERA TEST")
print("=" * 60)
print("👉 Look at the camera")
print("👉 Try different distances and angles")
print("👉 Press 'q' to quit")
print("=" * 60)

frame_count = 0
detection_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Cannot read frame")
        break
    
    frame_count += 1
    
    # Flip frame
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try multiple detection parameters
    # VERY RELAXED parameters to detect any face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # Lower = more sensitive
        minNeighbors=3,       # Lower = more detections (but more false positives)
        minSize=(30, 30),     # Smaller minimum face size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        detection_count += 1
    
    # Draw all detected faces
    for i, (x, y, w, h) in enumerate(faces):
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Label
        label = f"Face {i+1} ({w}x{h})"
        cv2.putText(frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Face area
        area = w * h
        cv2.putText(frame, f"Area: {area}", (x, y + h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Display statistics
    cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
    cv2.putText(frame, f"Detection rate: {detection_rate:.1f}%", (10, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Status message
    if len(faces) == 0:
        cv2.putText(frame, "NO FACE DETECTED", (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"DETECTING {len(faces)} FACE(S)!", (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Show original and grayscale side by side
    gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([frame, gray_3channel])
    
    cv2.imshow("Camera Test (Color | Grayscale)", combined)
    
    # Print to console every 30 frames
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}: Detected {len(faces)} face(s) | Rate: {detection_rate:.1f}%")
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Total frames: {frame_count}")
print(f"Frames with detection: {detection_count}")
print(f"Detection rate: {detection_rate:.1f}%")
print("=" * 60)

if detection_rate < 10:
    print("\n⚠️  LOW DETECTION RATE - TROUBLESHOOTING:")
    print("1. Check lighting - face should be well-lit")
    print("2. Face the camera directly")
    print("3. Remove glasses/hats if wearing")
    print("4. Move closer to the camera")
    print("5. Ensure camera is not covered or dirty")
    print("6. Try a different camera if available")
elif detection_rate < 50:
    print("\n⚠️  MODERATE DETECTION - TIPS:")
    print("1. Improve lighting conditions")
    print("2. Position face directly toward camera")
    print("3. Avoid extreme angles")
else:
    print("\n✅ GOOD DETECTION RATE!")
    print("Your camera and face detection are working properly.")
    print("You can now use the main registration system.")