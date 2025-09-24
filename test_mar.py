import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Initialize dlib's detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks (1).dat')

cap = cv2.VideoCapture(0)

print("Testing Mouth Aspect Ratio (MAR) values...")
print("Press 'q' to quit")
print("Try yawning to see MAR values increase")

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = np.array([(p.x, p.y) for p in landmarks.parts()])

        # Draw face rectangle
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate MAR
        mouth = landmarks_points[48:68]
        mar = mouth_aspect_ratio(mouth)
        
        # Draw mouth contour
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(img, [mouthHull], -1, (0, 255, 0), 1)
        
        # Display MAR value
        cv2.putText(img, f'MAR: {mar:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show threshold line
        cv2.putText(img, f'Threshold: 0.5', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Highlight if yawning detected
        if mar > 0.5:
            cv2.putText(img, "YAWNING DETECTED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"MAR: {mar:.3f} - Yawning detected!")

    cv2.imshow("MAR Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
