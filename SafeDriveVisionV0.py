import cv2
import dlib
import numpy as np
import torch
import math
import time
import pygame
import os
from scipy.spatial import distance as dist
from scipy.spatial import Delaunay
import threading

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

pygame.mixer.init()
current_time = time.time()

# Paths to audio files and associated delays
sounds = {
    'eye': ('./eyes_open.wav', 10),
    'look': ('./look_ahead.wav', 10),
    'phone': ('./no_phone.wav', 15),
    'alert': ('./stay_alert.wav', 10),
    'welcome': ('./welcomeengl.mp3', 0),
    'welcome_eng': ('./welcomeengl.mp3', 0)
}

# Last time the sound was played
last_played = {key: 0 for key in sounds}

def play_sound(sound_key):
    audio_file, delay = sounds[sound_key]
    current_time = time.time()
    if current_time - last_played[sound_key] > delay:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        last_played[sound_key] = current_time  # Update timestamp after playback

def sound_thread(sound_key):
    thread = threading.Thread(target=play_sound, args=(sound_key,))
    thread.daemon = True
    thread.start()

print("[INFO] project developed by: RMA Moroccan Insurance")

# Initialize dlib's detector and landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()  # HOG + SVM-based face detector
predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks (1).dat')  # Pre-trained facial landmark model

print("[INFO] initializing camera...")
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("testing_video.mp4")

# Control FPS of camera feed
desired_fps = 30
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# 3D reference model points of the face (nose, eyes, mouth, chin)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (-30.0, -125.0, -30.0), # Left eye corner
    (30.0, -125.0, -30.0),  # Right eye corner
    (-60.0, -70.0, -60.0),  # Left mouth corner
    (60.0, -70.0, -60.0),   # Right mouth corner
    (0.0, -330.0, -65.0)    # Chin
])

# Check if a rotation matrix is valid
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Convert rotation matrix to Euler angles (roll, pitch, yaw)
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])  # Pitch
        y = math.atan2(-R[2, 0], sy)      # Yaw
        z = math.atan2(R[1, 0], R[0, 0])  # Roll
    else:  # Gimbal lock case
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

# ### OPTIMIZATION ###: Modified this function to accept a pre-calculated camera_matrix
# This avoids recalculating it on every frame.
def getHeadTiltAndCoords(size, image_points, frame_height, camera_matrix):
    dist_coeffs = np.zeros((4, 1))
    
    # Estimate rotation and translation vectors
    (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                              camera_matrix, dist_coeffs,
                                                              flags=cv2.SOLVEPNP_ITERATIVE)
    # Project nose direction line
    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),
                                              rotation_vector, translation_vector,
                                              camera_matrix, dist_coeffs)

    # Convert to rotation matrix and Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    head_tilt_degree = abs([-180] - np.rad2deg([rotationMatrixToEulerAngles(rotation_matrix)[0]]))
    
    starting_point = (int(image_points[0][0]), int(image_points[0][1]))
    ending_point = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    ending_point_alternate = (ending_point[0], frame_height // 2)

    return head_tilt_degree, starting_point, ending_point, ending_point_alternate

# Eye Aspect Ratio (EAR) -> drowsiness detection
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Mouth Aspect Ratio (MAR) -> yawning detection
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Nose Aspect Ratio (experimental feature for head depth/position)
def nose_aspect_ratio(nose):
    vertical_distance = dist.euclidean(nose[0], nose[2])
    depth_distance = dist.euclidean(nose[0], nose[1])
    return depth_distance / vertical_distance

# Angle of head based on eyes and nose
def calculate_head_angle(eye_left, eye_right, nose_tip):
    eye_center = (eye_left + eye_right) / 2
    vector_nose = nose_tip - eye_center
    vector_horizontal = (eye_right - eye_left)
    vector_horizontal[1] = 0  # Flatten to horizontal axis
    vector_nose_normalized = vector_nose / np.linalg.norm(vector_nose)
    vector_horizontal_normalized = vector_horizontal / np.linalg.norm(vector_horizontal)
    angle_rad = np.arccos(np.clip(np.dot(vector_nose_normalized, vector_horizontal_normalized), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Load YOLOv5 object detection model
weights_path = os.path.join(os.path.dirname(__file__), 'yolov5m.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] YOLOv5 using device: {device}")
model.to(device)

# Variables to track alerts
COUNTER1 = 0  # Eye closure counter
COUNTER2 = 0  # Phone usage counter
COUNTER3 = 0  # Head angle counter
EYE_AR_CONSEC_FRAMES = 30
repeat_counter = 0
face_detected = False

# ### OPTIMIZATION ###: Calculate camera matrix once before the loop
# ret, img = cap.read()
# if not ret:
#     print("Failed to read video. Exiting.")
#     exit()
# camera_matrix = np.array([[img.shape[1], 0, img.shape[1]/2], 
#                           [0, img.shape[1], img.shape[0]/2], 
#                           [0, 0, 1]], dtype="double")

# Initialize camera matrix as None before the loop
camera_matrix = None

# ### OPTIMIZATION ###: Variables for frame skipping
frame_counter = 0
SKIP_FRAMES = 5  # Process every 5th frame
last_faces = []
last_detections = None

# Play welcome sounds
sound_thread('welcome')
sound_thread('welcome_eng')

while True:
    ret, img = cap.read()
    if not ret:
        print("Video ended, looping back to the start...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # if not ret:
    #     # For a live camera, this might mean it was disconnected.
    #     print("Failed to get frame. Exiting...")
    #     break
    
    # ### NEW CODE ###: Calculate camera matrix on the first successful frame
    if camera_matrix is None:
        camera_matrix = np.array([[img.shape[1], 0, img.shape[1]/2],
                                  [0, img.shape[1], img.shape[0]/2],
                                  [0, 0, 1]], dtype="double")

    frame_counter += 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale once

    # ### OPTIMIZATION ###: Run heavy detectors only on specified frames
    if frame_counter % SKIP_FRAMES == 0:
        last_faces = detector(gray, 0)  # Detect faces
        results = model(img)            # Object detection with YOLOv5
        last_detections = results.xyxy[0]

    # No face detected -> driver not looking
    if len(last_faces) == 0:
        cv2.putText(img, "The driver is not looking ahead", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
        sound_thread("look")

    # Use the last known detections for drawing on every frame
    if last_detections is not None:
        for detection in last_detections:
            if int(detection[5]) == 67:  # Class index for 'cell phone'
                x1, y1, x2, y2, conf = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]), detection[4]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(img, f'Cell Phone {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                COUNTER2 += 1
                if COUNTER2 >= 3 * SKIP_FRAMES: # Adjust counter threshold for skipped frames
                    cv2.putText(img, "Put away your CELL PHONE!", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    sound_thread("phone")
                    COUNTER2 = 0

    # Use the last known faces for drawing on every frame
    for face in last_faces:
        landmarks = predictor(gray, face)
        landmarks_points = np.array([(p.x, p.y) for p in landmarks.parts()])
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        image_points = np.array([
            landmarks_points[30], landmarks_points[36], landmarks_points[45],
            landmarks_points[48], landmarks_points[54], landmarks_points[8]
        ], dtype="double")

        # Eyes
        left_eye = landmarks_points[36:42]
        right_eye = landmarks_points[42:48]
        left_eyeHull = cv2.convexHull(left_eye)
        right_eyeHull = cv2.convexHull(right_eye)
        cv2.drawContours(img, [left_eyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(img, [right_eyeHull], -1, (255, 255, 255), 1)

        # ### BUG FIX ###: Corrected the EAR average calculation
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        # Mouth
        mouth = landmarks_points[48:68]
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(img, [mouthHull], -1, (0, 255, 0), 1)
        mar = mouth_aspect_ratio(mouth)

        # Display values
        cv2.putText(img, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(img, f'MAR: {mar:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Head angle and tilt
        head_angle = calculate_head_angle(np.array(landmarks_points[36]), np.array(landmarks_points[45]), np.array(landmarks_points[33]))
        cv2.putText(img, f'Head Angle: {head_angle:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        head_tilt_degree, start_point, end_point, end_point_alt = getHeadTiltAndCoords(img.shape, image_points, img.shape[0], camera_matrix)
        cv2.putText(img, f'Head Tilt: {head_tilt_degree[0]:.2f} degrees', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.line(img, start_point, end_point, (0, 255, 0), 2)

        # Alerts Logic
        if head_angle < 75 or head_angle > 110:
            COUNTER3 += 1
            if COUNTER3 >= 6 * SKIP_FRAMES:
                cv2.putText(img, "Look ahead!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                sound_thread("look")
                COUNTER3 = 0
        else:
            COUNTER3 = 0

        if ear < 0.33:
            COUNTER1 += 1
            if COUNTER1 >= 4 * SKIP_FRAMES:
                cv2.putText(img, "Eyes Closed!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                sound_thread("eye")
                repeat_counter += 1
                COUNTER1 = 0
                if repeat_counter >= 3:
                    sound_thread("alert")
                    repeat_counter = 0
        else:
            COUNTER1 = 0
            repeat_counter = 0

        if mar > 0.5:
            cv2.putText(img, "Yawning!", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            sound_thread("alert")

    # Resize the final frame with all drawings for display
    display_frame = cv2.resize(img, None, fx=0.75, fy=0.75)
    
    # cv2.imshow("Video Stream", img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit with 'q'
    #     break
    
    cv2.imshow("Frame", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()