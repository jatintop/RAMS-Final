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
# This dictionary will hold all the real-time data for the UI
system_status = {
    "attention_score": 100,
    "alertness": "High",
    "current_alert": "No Active Alert",
    "alert_timestamp": "",
    "alert_severity": "",
    "alert_history": [],
    "camera_connected": False
}
# A lock to ensure thread-safe access to the system_status dictionary
data_lock = threading.Lock()
# Global variable to hold the latest processed camera frame
output_frame = None
frame_lock = threading.Lock()
# Global flag to control the CV thread
cv_thread_running = threading.Event()

# --- Your Existing CV and Audio Logic (with slight modifications) ---

pygame.mixer.init()

sounds = {
    'eye': ('./eyes_open.wav', 10),
    'look': ('./look_ahead.wav', 10),
    'phone': ('./no_phone.wav', 15),
    'alert': ('./stay_alert.wav', 10),
    'welcome': ('./welcomeengl.mp3', 0)
}
last_played = {key: 0 for key in sounds}

def play_sound(sound_key):
    try:
        audio_file, delay = sounds[sound_key]
        current_time_sound = time.time()
        if current_time_sound - last_played[sound_key] > delay:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            last_played[sound_key] = current_time_sound
    except Exception as e:
        print(f"Error playing sound {sound_key}: {e}")

def sound_thread(sound_key):
    thread = threading.Thread(target=play_sound, args=(sound_key,))
    thread.daemon = True
    thread.start()

# --- dlib and YOLOv5 Model Loading ---
print("[INFO] Loading models...")
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks (1).dat')
    weights_path = './yolov5m.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("[INFO] Models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load models: {e}")
    exit()

# --- Your Helper Functions (isRotationMatrix, rotationMatrixToEulerAngles, etc.) ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# --- Main Computer Vision Logic Function (to be run in a thread) ---
# --- Main Computer Vision Logic Function (to be run in a thread) ---
def run_cv_logic():
    global system_status, output_frame, frame_lock, cv_thread_running

    print("[INFO] Initializing video source...")
    # --- LIVE CAMERA CODE (Commented out) ---
    # cap = cv2.VideoCapture(0)
    
    # --- VIDEO FILE CODE (Active) ---
    cap = cv2.VideoCapture("testing_video.mp4")
    
    time.sleep(2.0) # Give camera time to initialize
    
    # --- FRAME RATE CONTROL ---: Get the video's original FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # If FPS is not available, default to a standard rate
        fps = 25.0
    # Calculate the target time for each frame in seconds
    target_time_per_frame = 1 / fps
    
    with data_lock:
        system_status["camera_connected"] = cap.isOpened()

    # --- Constants and Counters ---
    EYE_AR_THRESH = 0.30
    EYE_AR_CONSEC_FRAMES = 10 # Frames for closed eyes
    YAWN_THRESH = 0.5
    PHONE_CONSEC_FRAMES = 5 # Frames for phone usage
    DISTRACTION_CONSEC_FRAMES = 20 # Frames for no face detected
    
    eye_counter = 0
    yawn_counter = 0
    phone_counter = 0
    distraction_counter = 0
    
    # --- OPTIMIZATION: Variables for frame skipping & processing smaller images ---
    frame_counter = 0
    SKIP_FRAMES = 5  # Process every 5th frame for major speed boost
    last_faces = []
    last_detections = None
    
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
            print("Video ended, looping back to the start...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_counter += 1
        
        # --- OPTIMIZATION: Create a smaller frame for faster processing ---
        original_h, original_w = frame.shape[:2]
        processing_w = 640
        scale_ratio = processing_w / original_w
        processing_h = int(original_h * scale_ratio)
        small_frame = cv2.resize(frame, (processing_w, processing_h))
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # --- OPTIMIZATION: Run heavy detectors only every SKIP_FRAMES ---
        if frame_counter % SKIP_FRAMES == 0:
            last_faces = detector(gray_small, 0)
            results = model(small_frame)
            last_detections = results.xyxy[0]

        # --- Local variables for this frame's analysis ---
        is_drowsy, is_yawning, is_distracted, using_phone = False, False, False, False
        face_detected_in_frame = len(last_faces) > 0
        current_attention_score = system_status["attention_score"]

        if face_detected_in_frame:
            distraction_counter = 0 # Reset distraction counter if face is seen
            if current_attention_score < 100: current_attention_score += 0.25

            for face in last_faces:
                # --- OPTIMIZATION: Scale landmark points from small frame to original frame ---
                landmarks = predictor(gray_small, face)
                landmarks_points = np.array([(int(p.x / scale_ratio), int(p.y / scale_ratio)) for p in landmarks.parts()])
                
                # Scale face bounding box
                x = int(face.left() / scale_ratio)
                y = int(face.top() / scale_ratio)
                w = int(face.width() / scale_ratio)
                h = int(face.height() / scale_ratio)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Process landmarks (EAR, MAR)
                left_ear = eye_aspect_ratio(landmarks_points[36:42])
                right_ear = eye_aspect_ratio(landmarks_points[42:48])
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(landmarks_points[48:68])
                
                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if ear < EYE_AR_THRESH:
                    eye_counter += 1
                    if eye_counter >= EYE_AR_CONSEC_FRAMES:
                        is_drowsy = True
                else:
                    eye_counter = 0

                if mar > YAWN_THRESH:
                    yawn_counter += 1
                    if yawn_counter > 2: is_yawning = True
                else:
                    yawn_counter = 0
        else:
            # No face detected
            distraction_counter += 1
            if distraction_counter >= DISTRACTION_CONSEC_FRAMES:
                is_distracted = True

        # --- YOLOv5 Phone Detection ---
        phone_detected_this_frame = False
        if last_detections is not None:
            for detection in last_detections:
                if int(detection[5]) == 67: # Class index for 'cell phone'
                    phone_detected_this_frame = True
                    phone_counter += 1
                    if phone_counter >= PHONE_CONSEC_FRAMES:
                        using_phone = True
                    
                    # --- OPTIMIZATION: Scale bounding box from small to original frame ---
                    x1 = int(detection[0] / scale_ratio)
                    y1 = int(detection[1] / scale_ratio)
                    x2 = int(detection[2] / scale_ratio)
                    y2 = int(detection[3] / scale_ratio)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break
        
        if not phone_detected_this_frame:
            phone_counter = 0 # Reset counter if no phone is seen

        # --- Update Shared System Status ---
        alert_msg, alert_sev, alertness_level = "No Active Alert", "", "High"
        
        if is_drowsy or is_yawning:
            alert_msg, alert_sev, alertness_level = "Drowsiness Detected", "HIGH", "Low"
            current_attention_score -= 2
            if is_drowsy: play_sound('eye')
            if is_yawning: play_sound('alert')
            cv2.putText(frame, "DROWSINESS ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif using_phone:
            alert_msg, alert_sev, alertness_level = "Phone Usage Detected", "MEDIUM", "Medium"
            current_attention_score -= 1
            play_sound('phone')
            cv2.putText(frame, "NO PHONE!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif is_distracted:
            alert_msg, alert_sev, alertness_level = "Distraction Detected", "LOW", "Medium"
            current_attention_score -= 0.5
            play_sound('look')
            cv2.putText(frame, "LOOK AHEAD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        current_attention_score = max(0, min(100, current_attention_score))

        with data_lock:
            if alert_msg != "No Active Alert" and alert_msg != system_status.get("last_alert_msg"):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                system_status["alert_timestamp"] = timestamp
                system_status["alert_history"].insert(0, {"alert": alert_msg, "severity": alert_sev, "time": timestamp})
                if len(system_status["alert_history"]) > 10: system_status["alert_history"].pop()
            
            system_status["last_alert_msg"] = alert_msg if alert_msg != "No Active Alert" else system_status.get("last_alert_msg")
            system_status["current_alert"] = alert_msg
            system_status["alert_severity"] = alert_sev
            system_status["alertness"] = alertness_level
            system_status["attention_score"] = int(current_attention_score)
            system_status["camera_connected"] = True
            
        with frame_lock:
            cv2.putText(frame, "Driver Monitoring: Active", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
    """Video streaming home page."""
    return render_template('index.html')

def gen_frames():
    """Generator function for video streaming."""
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                # If no frame, could show a "disconnected" image
                # For now, just skip
                time.sleep(0.1) # prevent busy-waiting
                continue
            frame = output_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """API endpoint to get the current system status."""
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
        # cv_thread.join() # This can cause the UI to hang
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
    """Clears the alert history."""
    with data_lock:
        system_status["alert_history"] = []
    return jsonify({"status": "success", "message": "Alert history cleared."})

# --- Main Execution ---
if __name__ == '__main__':
    # The CV thread is now started via a Flask route, not automatically
    app.run(host='0.0.0.0', port=8501, debug=False)