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
    mar = (A + B) / (2.0 * C)
    return mar

# --- Main Computer Vision Logic Function (to be run in a thread) ---
def run_cv_logic():
    global system_status, output_frame, frame_lock, cv_thread_running
    if not all([detector, predictor, model]):
        print("[ERROR] CV/YOLO models failed to load. Cannot run CV logic.")
        return

    print("[INFO] Initializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(2.0)
    
    with data_lock:
        system_status["camera_connected"] = cap.isOpened()

    # Counters and constants
    EYE_AR_THRESH = 0.25 # Threshold for eye closure
    EYE_AR_CONSEC_FRAMES = 5 # Number of consecutive frames eyes must be closed
    YAWN_THRESH = 0.5
    
    eye_counter = 0
    yawn_counter = 0
    distraction_counter = 0
    
    play_sound('welcome')

    while cv_thread_running.is_set():
         # --- FRAME RATE CONTROL ---: Record the start time of the loop
        start_time = time.time()
        
        if not cap.isOpened():
            with data_lock: system_status["camera_connected"] = False
            time.sleep(1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # --- Local variables for this frame's analysis ---
        is_drowsy = False
        is_yawning = False
        is_distracted = False
        using_phone = False
        face_detected_in_frame = False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        
        # Update UI data: reset if no issues
        current_attention_score = system_status["attention_score"]

        if len(faces) > 0:
            face_detected_in_frame = True
            # Recover attention score if driver is attentive
            if current_attention_score < 100:
                 current_attention_score += 0.25
            
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                landmarks = predictor(gray, face)
                landmarks_points = np.array([(p.x, p.y) for p in landmarks.parts()])
                
                left_eye = landmarks_points[36:42]
                right_eye = landmarks_points[42:48]
                mouth = landmarks_points[48:68]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(mouth)

                # Drowsiness detection (closed eyes)
                if ear < EYE_AR_THRESH:
                    drowsy_counter += 1
                    if drowsy_counter >= 3: 
                        is_drowsy = True
                        play_sound('eye')
                        cv2.putText(frame, "DROWSINESS ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    drowsy_counter = max(0, drowsy_counter - 1)

                # Yawning detection
                if mar > YAWN_THRESH:
                    yawn_counter += 1
                    if yawn_counter > 2: # Check for a couple of frames to be sure
                        is_yawning = True
                        play_sound('alert')
                        cv2.putText(frame, "YAWNING ALERT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    yawn_counter = max(0, yawn_counter - 1)
                    
            cv2.putText(frame, f'Yaw: {yaw:.1f} | EAR: {ear:.2f} | MAR: {mar:.2f}', (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            # No face detected
            is_distracted = True
            distraction_counter += 1
            if distraction_counter > 30: # If no face for about a second
                play_sound('look')
                cv2.putText(frame, "LOOK AHEAD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                distraction_counter = 0
        
        # --- YOLOv5 Phone Detection ---
        results = model(frame)
        detections = results.xyxy[0]
        for detection in detections:
            if int(detection[5]) == 67: # Class index for 'cell phone'
                using_phone = True
                phone_counter +=1
                if phone_counter > 15: # If phone detected for half a second
                    play_sound('phone')
                    cv2.putText(frame, "NO PHONE!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    phone_counter = 0
                break # only need one phone detection
        
        # --- Update Shared System Status ---
        alert_msg = "No Active Alert"
        alert_sev = ""
        alertness_level = "High"

        # Prioritize alerts
        if is_drowsy or is_yawning:
            alert_msg = "Drowsiness Detected"
            alert_sev = "HIGH"
            alertness_level = "Low"
            current_attention_score -= 2 # Penalize heavily
        elif using_phone:
            alert_msg = "Phone Usage Detected"
            alert_sev = "MEDIUM"
            alertness_level = "Medium"
            current_attention_score -= 1 # Penalize moderately
        elif is_distracted and not face_detected_in_frame:
            alert_msg = "Distraction Detected"
            alert_sev = "LOW"
            alertness_level = "Medium"
            current_attention_score -= 0.5 # Penalize lightly

        # Clamp attention score between 0 and 100
        current_attention_score = max(0, min(100, current_attention_score))

        # Update the global status dictionary
        with data_lock:
            if alert_msg != system_status["current_alert"] and alert_msg != "No Active Alert":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                system_status["alert_timestamp"] = timestamp
                # Add to history (and keep history size manageable)
                system_status["alert_history"].insert(0, {
                    "alert": alert_msg, "severity": alert_sev, "time": timestamp
                })
                if len(system_status["alert_history"]) > 10:
                    system_status["alert_history"].pop()
            
            system_status["current_alert"] = alert_msg
            system_status["alert_severity"] = alert_sev
            system_status["alertness"] = alertness_level
            system_status["attention_score"] = int(current_attention_score)
            system_status["camera_connected"] = True
            
        # Update the global frame for streaming
        with frame_lock:
            # Draw final monitoring status on the frame
            status_color = (0, 255, 0) if current_alert_msg == "No Active Alert" else (0, 0, 255)
            cv2.putText(frame, "Driver Monitoring: Active", (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            output_frame = buffer.tobytes()
            
        # --- FRAME RATE CONTROL ---: Calculate elapsed time and sleep if necessary
        elapsed_time = time.time() - start_time
        sleep_time = target_time_per_frame - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    print("[INFO] CV thread has stopped and released the video source.")

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
        return jsonify(system_status)

@app.route('/start_cv', methods=['POST'])
def start_cv():
    """Starts the computer vision processing thread."""
    global cv_thread
    if not cv_thread_running.is_set():
        cv_thread_running.set()
        cv_thread = threading.Thread(target=run_cv_logic)
        cv_thread.daemon = True
        cv_thread.start()
        return jsonify({"status": "success", "message": "CV thread started."})
    return jsonify({"status": "info", "message": "CV thread is already running."})

@app.route('/stop_cv', methods=['POST'])
def stop_cv():
    """Stops the computer vision processing thread."""
    global output_frame
    if cv_thread_running.is_set():
        cv_thread_running.clear()
        # cv_thread.join() # This can cause a delay
        with frame_lock:
            output_frame = None # Clear the last frame to black out the feed
        with data_lock:
            system_status["camera_connected"] = False
            system_status["current_alert"] = "No Active Alert"
            system_status["alertness"] = "N/A"
            system_status["attention_score"] = 0
        return jsonify({"status": "success", "message": "CV thread stopped."})
    return jsonify({"status": "info", "message": "CV thread is not running."})

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