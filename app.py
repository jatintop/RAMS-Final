# app.py

import cv2
import dlib
import numpy as np
import torch
import math
import time
import pygame 
import os
import threading
import datetime
from scipy.spatial import distance as dist
from flask import Flask, render_template, Response, jsonify

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Global variables & thread lock for sharing data ---
system_status = {
    "attention_score": 100, # Initial score is 100
    "alertness": "High",
    "current_alert": "No Active Alert",
    "alert_timestamp": "",
    "alert_severity": "",
    "alert_history": [],
    "camera_connected": False
}
data_lock = threading.Lock()
output_frame = None
frame_lock = threading.Lock()
cv_thread_running = threading.Event()
cv_thread = None

# --- CV and Audio Logic Initialization ---

# Initialize Pygame Mixer for sound
try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"[WARNING] Could not initialize Pygame Mixer: {e}. Audio alerts will not work.")

# NOTE: Ensure these audio files exist in the same directory as app.py
sounds = {
    'eye': ('./eyes_open.wav', 10),
    'look': ('./look_ahead.wav', 10),
    'phone': ('./no_phone.wav', 5),  # Reduced delay for phone alert
    'alert': ('./stay_alert.wav', 10),
    'welcome': ('./welcomeengl.mp3', 0),
}
last_played = {key: 0 for key in sounds}

# Initialize dlib's detector and landmark predictor
try:
    print("[INFO] Loading CV Models...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks (1).dat') 
    
    # Load YOLOv5 model once
    weights_path = os.path.join(os.path.dirname(__file__), 'yolov5m.pt') 
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print("[INFO] CV Models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load CV models: {e}. Check your model file paths.")
    detector, predictor, model = None, None, None

# 3D reference model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),      # Nose tip (30)
    (-30.0, -125.0, -30.0), # Left eye corner (36)
    (30.0, -125.0, -30.0),  # Right eye corner (45)
    (-60.0, -70.0, -60.0),  # Left mouth corner (48)
    (60.0, -70.0, -60.0),  # Right mouth corner (54)
    (0.0, -330.0, -65.0)    # Chin (8)
])

# --- Helper Functions ---

def play_sound(sound_key):
    try:
        audio_file, delay = sounds[sound_key]
        current_time_sound = time.time()
        if current_time_sound - last_played[sound_key] > delay:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                last_played[sound_key] = current_time_sound
    except Exception:
        pass 

def sound_thread(sound_key):
    thread = threading.Thread(target=play_sound, args=(sound_key,))
    thread.daemon = True
    thread.start()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def get_head_yaw(size, image_points):
    """Calculates the head's yaw (left-right rotation)."""
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    
    # Check if we have enough points (6 in this case)
    if len(image_points) < 6:
        return 0.0
        
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if success:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.degrees(yaw)
    return 0.0

def update_alert_status(msg, severity, alertness, score_penalty):
    """
    Utility to update the global system_status state and log alerts.
    Log is now persistent until manually cleared.
    """
    global system_status
    current_time_str = datetime.datetime.now().strftime("%H:%M:%S")

    with data_lock:
        current_score = system_status["attention_score"]

        # 1. Base Score Update (Recovery vs. Penalty)
        RECOVERY_RATE = 1.0  # Increased for noticeable recovery
        
        if msg == "No Active Alert":
            # RECOVERY
            new_score = current_score + RECOVERY_RATE
            
            # Determine alertness based on the recovering score
            if new_score > 75: alertness = "High"
            elif new_score > 30: alertness = "Medium"
            else: alertness = "Low"

        else:
            # PENALTY
            new_score = current_score - score_penalty
            
        # Clamp score between 0 and 100
        system_status["attention_score"] = max(0, min(100, int(new_score)))
        
        # 2. Update Alert Log and Current Alert Status
        if msg != "No Active Alert":
            # Log only if it's a new alert or a persistent critical alert
            if msg != system_status["current_alert"] or severity == "HIGH":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                system_status["alert_history"].insert(0, {"alert": msg, "severity": severity, "time": timestamp})
                
                # NOTE: Log limit (e.g., pop() after 15 entries) REMOVED to keep all records.
            
            # Set the current active alert
            system_status["current_alert"] = msg
            system_status["alert_severity"] = severity
            system_status["alert_timestamp"] = current_time_str
            system_status["alertness"] = alertness # Use the alertness passed by the CV logic
        else:
            # Clear current alert only if we are in recovery mode
            system_status["current_alert"] = "No Active Alert"
            system_status["alert_severity"] = ""
            system_status["alertness"] = alertness # Use the alertness determined by the score


# --- Main Computer Vision Logic Function ---
def run_cv_logic():
    global system_status, output_frame, frame_lock, cv_thread_running
    if not all([detector, predictor, model]):
        print("[ERROR] CV/YOLO models failed to load. Cannot run CV logic.")
        return

    cap = cv2.VideoCapture(0)
    time.sleep(1.0)
    
    with data_lock:
        system_status["camera_connected"] = cap.isOpened()

    # CV Constants and Counters
    EYE_AR_THRESH = 0.25
    MAR_THRESH = 0.50 
    YAW_THRESH = 15  
    YOLO_SKIP_FRAMES = 10 
    YOLO_CONFIDENCE_THRESHOLD = 0.3 # Lowered for better detection of side-held phones

    drowsy_counter = 0    
    yawn_counter = 0
    distraction_counter = 0
    frame_count = 0 
    phone_detected_this_cycle = False 

    sound_thread('welcome') 

    while cv_thread_running.is_set():
        if not cap.isOpened():
            with data_lock: system_status["camera_connected"] = False
            time.sleep(1)
            continue
            
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) 
        frame_count += 1
        (h, w) = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        
        # Reset alert flags for the current frame
        is_drowsy, is_yawning, is_distracted_face, is_distracted_no_face = False, False, False, False
        
        # 1. Facial and Pose Analysis 
        if len(faces) > 0:
            
            for face in faces:
                x, y, w_f, h_f = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), (0, 255, 0), 2)
                landmarks_points = np.array([(p.x, p.y) for p in predictor(gray, face).parts()])

                # Head Pose
                image_points = np.array([
                    landmarks_points[30], landmarks_points[36], landmarks_points[45], 
                    landmarks_points[48], landmarks_points[54], landmarks_points[8] 
                ], dtype="double")
                yaw = get_head_yaw(frame.shape, image_points)
                
                # Head Yaw Check (Distraction)
                if abs(yaw) > YAW_THRESH:
                    is_distracted_face = True
                    distraction_counter += 1
                    cv2.putText(frame, "DISTRACTED: Yaw", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    distraction_counter = max(0, distraction_counter - 1)
                
                # EAR (Drowsiness/Eyes)
                left_ear = eye_aspect_ratio(landmarks_points[36:42])
                right_ear = eye_aspect_ratio(landmarks_points[42:48])
                ear = (left_ear + right_ear) / 2.0
                if ear < EYE_AR_THRESH:
                    drowsy_counter += 1
                    if drowsy_counter >= 3: 
                        is_drowsy = True
                        cv2.putText(frame, "DROWSY: Eyes Closed", (x, y + h_f + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    drowsy_counter = max(0, drowsy_counter - 1)

                # MAR (Yawning)
                mar = mouth_aspect_ratio(landmarks_points[48:68])
                if mar > MAR_THRESH:
                    yawn_counter += 1
                    if yawn_counter >= 2:
                        is_yawning = True
                        cv2.putText(frame, "DROWSY: Yawning", (x, y + h_f + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    yawn_counter = max(0, yawn_counter - 1)
                    
            cv2.putText(frame, f'Yaw: {yaw:.1f} | EAR: {ear:.2f} | MAR: {mar:.2f}', (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            # No face detected
            is_distracted_no_face = True
            distraction_counter += 1
            cv2.putText(frame, "DISTRACTED: Face not visible", (w // 2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        
        # 2. YOLOv5 Phone Detection 
        if frame_count % YOLO_SKIP_FRAMES == 0: 
            phone_detected_this_cycle = False 
            with torch.no_grad():
                 results = model(frame) 
            detections = results.xyxy[0]
            
            for detection in detections:
                if int(detection[5]) == 67 and detection[4] > YOLO_CONFIDENCE_THRESHOLD: 
                    phone_detected_this_cycle = True 
                    break 
            frame_count = 0
             
        # --- Instant Alert Flag for Phone ---
        is_using_phone = phone_detected_this_cycle
        
        if is_using_phone:
             cv2.putText(frame, "PHONE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 3. Aggregation and Alert Management
        current_alert_msg, alert_severity, alertness_level, score_penalty = "No Active Alert", "", "High", 0.0 

        # --- Determine the HIGHEST priority alert ---
        if is_drowsy or is_yawning:
            current_alert_msg, alert_severity, alertness_level, score_penalty = "Drowsiness Detected", "HIGH", "Low", 5.0 
            if is_drowsy and drowsy_counter > 5: sound_thread('eye')
            if is_yawning: sound_thread('alert')
            
        elif is_using_phone: # INSTANT ALERT
            current_alert_msg, alert_severity, alertness_level, score_penalty = "Phone Usage Detected", "HIGH", "Low", 4.0
            sound_thread('phone')
            
        elif is_distracted_face or (is_distracted_no_face and distraction_counter > 15):
            current_alert_msg, alert_severity, alertness_level, score_penalty = "Distraction Detected", "MEDIUM", "Medium", 2.0
            if distraction_counter > 20: sound_thread('look')

        # Update the shared status
        update_alert_status(current_alert_msg, alert_severity, alertness_level, score_penalty)
        
        # 4. Prepare Frame for Streaming
        with frame_lock:
            # Draw final monitoring status on the frame
            status_color = (0, 255, 0) if current_alert_msg == "No Active Alert" else (0, 0, 255)
            cv2.putText(frame, "Driver Monitoring: Active", (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            output_frame = buffer.tobytes()

    cap.release()

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    """Generator function to stream video frames."""
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.01) 
                continue
            frame = output_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """API endpoint for the UI to fetch real-time status."""
    with data_lock:
        return jsonify(system_status.copy())

@app.route('/clear_log', methods=['POST'])
def clear_log():
    """API endpoint to clear the alert history."""
    with data_lock:
        # This is the function that actually clears the persistent log
        system_status["alert_history"] = []
    return jsonify({"status": "success", "message": "Alert history cleared."})

# --- Main Execution ---
if __name__ == '__main__':
    # Ensure templates and static directories exist
    if not os.path.exists('templates'): os.makedirs('templates')
    if not os.path.exists('static'): os.makedirs('static')
    
    # Start the Computer Vision thread automatically
    if not cv_thread_running.is_set():
        cv_thread_running.set()
        cv_thread = threading.Thread(target=run_cv_logic)
        cv_thread.daemon = True
        cv_thread.start()
        print("[INFO] CV thread started automatically.")
    
    # Start the Flask web server
    app.run(host='0.0.0.0', port=8501, debug=False, threaded=True, use_reloader=False)