import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
from PIL import Image

# Configure page
st.set_page_config(
    page_title="Driver Assistance System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for blue glowing professional look
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0c1445 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #00d4ff;
        text-shadow: 0 0 20px #00d4ff, 0 0 40px #00d4ff, 0 0 60px #00d4ff;
        margin-bottom: 2rem;
        padding: 1rem;
        background: rgba(0, 212, 255, 0.1);
        border-radius: 15px;
        border: 2px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2);
    }
    
    .camera-container {
        background: rgba(0, 212, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
        margin-bottom: 1rem;
    }
    
    .active-alert-container {
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.2), rgba(255, 100, 100, 0.1));
        border: 2px solid #ff0000;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(255, 0, 0, 0.4);
        animation: glow-red 2s ease-in-out infinite alternate;
    }
    
    .alert-log-container {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .alert-log-item {
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #ffaa00;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    
    .alert-log-item.high {
        border-left-color: #ff4444;
        background: rgba(255, 68, 68, 0.1);
    }
    
    .alert-log-item.medium {
        border-left-color: #ffaa00;
        background: rgba(255, 170, 0, 0.1);
    }
    
    .alert-log-item.low {
        border-left-color: #ffff00;
        background: rgba(255, 255, 0, 0.1);
    }
    
    .alert-text {
        color: #ff4444;
        font-size: 1.3rem;
        font-weight: bold;
        text-shadow: 0 0 10px #ff4444;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .datetime-container {
        background: rgba(0, 212, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(0, 212, 255, 0.3);
        text-align: center;
        color: #00d4ff;
        font-size: 1.1rem;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
    }
    
    .status-normal {
        background: rgba(0, 255, 0, 0.1);
        border: 2px solid #00ff00;
        border-radius: 10px;
        padding: 1rem;
        color: #00ff88;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);
    }
    
    @keyframes glow-red {
        from { box-shadow: 0 0 20px rgba(255, 0, 0, 0.4); }
        to { box-shadow: 0 0 40px rgba(255, 0, 0, 0.8); }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(255, 0, 0, 0.5);
    }
    
    .metric-container {
        background: rgba(0, 212, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(0, 212, 255, 0.3);
        text-align: center;
    }
    
    .camera-status {
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .camera-connected {
        background: rgba(0, 255, 0, 0.2);
        border: 1px solid #00ff00;
        color: #00ff88;
    }
    
    .camera-disconnected {
        background: rgba(255, 0, 0, 0.2);
        border: 1px solid #ff0000;
        color: #ff4444;
    }

    .log-header {
        color: #00d4ff;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }

    .clear-log-btn > button {
        background: linear-gradient(135deg, #666666, #444444);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.3rem 1rem;
        font-size: 0.9rem;
        box-shadow: 0 0 10px rgba(100, 100, 100, 0.3);
    }

    .demo-control {
        background: rgba(0, 212, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(0, 212, 255, 0.3);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_alert' not in st.session_state:
    st.session_state.current_alert = None
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'demo_alert_shown' not in st.session_state:
    st.session_state.demo_alert_shown = False
if 'camera_initialized' not in st.session_state:
    st.session_state.camera_initialized = False
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Create demo alert for demonstration
def create_demo_alert():
    if not st.session_state.demo_alert_shown and not st.session_state.current_alert:
        demo_alert = {
            'type': "Drowsiness Detected",
            'timestamp': datetime.now(),
            'severity': 'HIGH',
            'id': f"demo_alert_{int(time.time())}"
        }
        set_alert(demo_alert)
        st.session_state.demo_alert_shown = True

# Set current alert and add to log
def set_alert(alert):
    st.session_state.current_alert = alert
    log_entry = {
        **alert,
        'status': 'Active',
        'dismissed_at': None
    }
    st.session_state.alert_log.insert(0, log_entry)

# Dismiss current alert
def dismiss_current_alert():
    if st.session_state.current_alert:
        for log_entry in st.session_state.alert_log:
            if log_entry['id'] == st.session_state.current_alert['id']:
                log_entry['status'] = 'Dismissed'
                log_entry['dismissed_at'] = datetime.now()
                break
        st.session_state.current_alert = None

# Initialize camera
def initialize_camera():
    if not st.session_state.camera_initialized:
        st.session_state.camera = cv2.VideoCapture(0)
        if not st.session_state.camera.isOpened():
            st.error("❌ Could not access the webcam.")
            st.session_state.camera_initialized = False
        else:
            st.session_state.camera_initialized = True

# Header
st.markdown('<div class="main-header">Real-Time Attention Monitoring System</div>', unsafe_allow_html=True)

# Create main layout
col1, col2 = st.columns([2, 1])

with col1:
    # Camera feed section
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    st.subheader("📹 Live Camera Feed")
    
    # Camera status
    if st.session_state.camera_initialized:
        st.markdown('<div class="camera-status camera-connected">🟢 Camera Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="camera-status camera-disconnected">🔴 Camera Disconnected</div>', unsafe_allow_html=True)
    
    # Camera feed placeholder
    FRAME_WINDOW = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Date and Time
    current_time = datetime.now()
    st.markdown(f'''
    <div class="datetime-container">
        <strong>📅 {current_time.strftime("%A, %B %d, %Y")}</strong><br>
        <strong>🕐 {current_time.strftime("%I:%M:%S %p")}</strong>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    # Current Alert section
    st.subheader("⚠️ Current Alert")
    
    if st.session_state.current_alert:
        alert = st.session_state.current_alert
        st.markdown(f'''
        <div class="active-alert-container">
            <div class="alert-text">🚨 {alert["type"]}</div>
            <div style="text-align: center; color: #ffaa00;">
                Time: {alert["timestamp"].strftime("%H:%M:%S")}<br>
                Severity: {alert["severity"]}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Dismiss button
        if st.button("✅ Dismiss Alert", key="dismiss_current", use_container_width=True):
            dismiss_current_alert()
            # No rerun here to prevent camera feed interruption
    else:
        st.markdown('''
        <div class="status-normal">
            <h4>✅ No Active Alert</h4>
            <p>System monitoring normally</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Alert Log section
    st.markdown("---")
    st.markdown('<div class="log-header">📋 Alert History Log</div>', unsafe_allow_html=True)
    
    # Clear log button and count
    col_clear, col_count = st.columns([1, 1])
    with col_clear:
        if st.button("🗑️ Clear Log", key="clear_log", help="Clear all alert history"):
            st.session_state.alert_log = []
            # No rerun here to prevent camera feed interruption
    
    with col_count:
        st.write(f"**Total: {len(st.session_state.alert_log)} alerts**")
    
    # Display alert log
    if st.session_state.alert_log:
        st.markdown('<div class="alert-log-container">', unsafe_allow_html=True)
        
        # Show only the first alert for demo purposes
        log_entry = st.session_state.alert_log[0]
        severity_class = log_entry['severity'].lower()
        status_color = "#00ff88" if log_entry['status'] == 'Dismissed' else "#ff4444"
        dismissed_text = ""
        if log_entry['dismissed_at']:
            dismissed_text = f"<br><small>Dismissed: {log_entry['dismissed_at'].strftime('%H:%M:%S')}</small>"
        
        st.markdown(f'''
        <div class="alert-log-item {severity_class}">
            <strong>{log_entry["type"]}</strong><br>
            <small>{log_entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</small><br>
            <small>Severity: {log_entry["severity"]} | Status: <span style="color: {status_color}">{log_entry["status"]}</span></small>
            {dismissed_text}
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No alerts in history")
    
    # System status metrics
    st.markdown("---")
    st.subheader("📊 System Status")
    
    col_a, col_b = st.columns(2)
    with col_a:
        # Calculate attention based on current alert
        attention_level = 85 if not st.session_state.current_alert else 65
        color = "#00ff88" if attention_level > 80 else "#ffaa00" if attention_level > 60 else "#ff4444"
        
        st.markdown(f'''
        <div class="metric-container">
            <h4 style="color: #00d4ff;">Attention</h4>
            <h2 style="color: {color};">{attention_level}%</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_b:
        alertness = "High" if not st.session_state.current_alert else "Low"
        color = "#00ff88" if alertness == "High" else "#ff4444"
        
        st.markdown(f'''
        <div class="metric-container">
            <h4 style="color: #00d4ff;">Alertness</h4>
            <h2 style="color: {color};">{alertness}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    # Demo control
    st.markdown("---")
    st.markdown(f'''
    <div class="demo-control">
        <h4 style="color: #00d4ff; text-align: center;">Demo Controls</h4>
    </div>
    ''', unsafe_allow_html=True)
    
    if st.button("🚨 Trigger Demo Alert", key="demo_alert", use_container_width=True):
        create_demo_alert()
        # No rerun here to prevent camera feed interruption

# Camera loop
initialize_camera()

if st.session_state.camera_initialized:
    while True:
        ret, frame = st.session_state.camera.read()
        if not ret:
            st.error("⚠️ Failed to read from camera.")
            break

        st.session_state.frame_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Overlay timestamp and frame count
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {st.session_state.frame_count}", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Driver Monitoring Active", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert BGR (OpenCV) to RGB (Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, caption="Live Webcam Feed", use_container_width=True)
        
        # Small sleep to prevent overwhelming the UI
        time.sleep(0.03)  # Approximately 30 FPS

    # Release camera if loop breaks
    if st.session_state.camera:
        st.session_state.camera.release()
        st.session_state.camera_initialized = False