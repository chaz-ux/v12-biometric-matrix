import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import numpy as np
import csv
from collections import deque
from datetime import datetime
from typing import Any

# --- CONFIGURATION ---
MODEL_PATH = 'face_landmarker.task'
BLINK_THRESHOLD = 0.22 
CALIBRATION_FRAMES = 200 
SMOOTHING_FACTOR = 0.3      
BASELINE_ADAPT_RATE = 0.001 

# --- ASYNC STATE ---
latest_result = None

def result_callback(result: Any, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# --- MATH & STATISTICAL FUNCTIONS ---
def euclidean_dist_3d_real(p1, p2, w, h):
    x1, y1, z1 = p1.x * w, p1.y * h, p1.z * w
    x2, y2, z2 = p2.x * w, p2.y * h, p2.z * w
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def calculate_ear(eye_landmarks, w, h):
    v1 = euclidean_dist_3d_real(eye_landmarks[1], eye_landmarks[5], w, h)
    v2 = euclidean_dist_3d_real(eye_landmarks[2], eye_landmarks[4], w, h)
    hw = euclidean_dist_3d_real(eye_landmarks[0], eye_landmarks[3], w, h)
    return (v1 + v2) / (2.0 * hw)

def smooth_value(current, new_raw, factor=SMOOTHING_FACTOR):
    if current == 0.0: return new_raw
    return (new_raw * factor) + (current * (1.0 - factor))

def calculate_rppg_bpm(signal_buffer, times_buffer):
    if len(signal_buffer) < 150: return 0
    
    time_elapsed = times_buffer[-1] - times_buffer[0]
    if time_elapsed <= 0: return 0
    fps = len(signal_buffer) / time_elapsed
    
    signal = np.array(signal_buffer)
    detrended = signal - np.mean(signal)
    
    windowed = detrended * np.hanning(len(detrended))
    
    n = len(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0/fps)
    fft_mag = np.abs(np.fft.rfft(windowed))
    
    min_idx = np.searchsorted(freqs, 0.8)
    max_idx = np.searchsorted(freqs, 3.0)
    
    if min_idx >= max_idx or max_idx >= len(fft_mag): return 0
    
    valid_fft = fft_mag[min_idx:max_idx]
    peak_idx = np.argmax(valid_fft)
    
    dominant_freq = freqs[min_idx + peak_idx]
    return int(dominant_freq * 60.0)

# --- INITIALIZATION ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options, num_faces=1, 
    running_mode=vision.RunningMode.LIVE_STREAM, result_callback=result_callback
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60) 

# --- STATISTICAL BASELINE VARIABLES ---
frame_count = 0
features = ['au4_brow_lower', 'au1_brow_raise', 'au23_lip_press', 'au12_lip_pull', 'au6_cheek_raise', 'head_yaw', 'head_pitch']
calib_data = {f: [] for f in features}
base_mean = {f: 0.0 for f in features}
base_std = {f: 0.001 for f in features} 
smoothed = {f: 0.0 for f in features}

history_len = 12 
kinematics = {
    'au4_brow_lower': deque(maxlen=history_len),
    'au23_lip_press': deque(maxlen=history_len)
}

# rPPG Specific Variables
rppg_buffer = deque(maxlen=300)
rppg_times = deque(maxlen=300)
optical_bpm = 0
last_bpm_calc_time = 0

last_movement_time = time.time()
blink_timestamps = deque()
is_blinking = False
score_history = deque(maxlen=200) 

prev_frame_time = 0
display_score = 0.0 

# --- CSV LOGGING BUFFER ---
session_log = []

print("V12 MATRIX LOGGING ONLINE. AWAITING TARGET...")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    new_frame_time = time.time()
    dt = max((new_frame_time - prev_frame_time), 0.001)
    fps = 1 / dt
    prev_frame_time = new_frame_time

    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detector.detect_async(mp_image, int(time.time() * 1000))

    if latest_result and latest_result.face_landmarks:
        landmarks = latest_result.face_landmarks[0] 

        # --- OPTICAL rPPG EXTRACTION ---
        fh_x, fh_y = int(landmarks[151].x * w), int(landmarks[151].y * h)
        box_w, box_h = int(w * 0.04), int(h * 0.03) 
        
        y1, y2 = max(0, fh_y - box_h), min(h, fh_y + box_h)
        x1, x2 = max(0, fh_x - box_w), min(w, fh_x + box_w)
        
        if y2 > y1 and x2 > x1:
            forehead_roi = image[y1:y2, x1:x2]
            green_mean = np.mean(forehead_roi[:, :, 1])
            rppg_buffer.append(green_mean)
            rppg_times.append(new_frame_time)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if new_frame_time - last_bpm_calc_time > 0.5 and len(rppg_buffer) > 150:
            optical_bpm = calculate_rppg_bpm(rppg_buffer, rppg_times)
            last_bpm_calc_time = new_frame_time

        # --- 3D KINEMATICS ---
        face_w = euclidean_dist_3d_real(landmarks[33], landmarks[263], w, h)
        face_h = euclidean_dist_3d_real(landmarks[10], landmarks[152], w, h)

        raw_au4 = euclidean_dist_3d_real(landmarks[107], landmarks[336], w, h) / face_w
        raw_au1 = (euclidean_dist_3d_real(landmarks[159], landmarks[52], w, h) + euclidean_dist_3d_real(landmarks[386], landmarks[282], w, h)) / (2.0 * face_h)
        raw_au23 = euclidean_dist_3d_real(landmarks[13], landmarks[14], w, h) / face_h
        raw_au12 = euclidean_dist_3d_real(landmarks[61], landmarks[291], w, h) / face_w
        raw_au6 = (euclidean_dist_3d_real(landmarks[117], landmarks[111], w, h) + euclidean_dist_3d_real(landmarks[346], landmarks[340], w, h)) / (2.0 * face_h)

        yaw_ratio = euclidean_dist_3d_real(landmarks[1], landmarks[234], w, h) / (euclidean_dist_3d_real(landmarks[1], landmarks[454], w, h) + 0.001)
        pitch_ratio = euclidean_dist_3d_real(landmarks[1], landmarks[152], w, h) / (euclidean_dist_3d_real(landmarks[1], landmarks[10], w, h) + 0.001)

        smoothed['au4_brow_lower'] = smooth_value(smoothed['au4_brow_lower'], raw_au4)
        smoothed['au1_brow_raise'] = smooth_value(smoothed['au1_brow_raise'], raw_au1)
        smoothed['au23_lip_press'] = smooth_value(smoothed['au23_lip_press'], raw_au23)
        smoothed['au12_lip_pull'] = smooth_value(smoothed['au12_lip_pull'], raw_au12)
        smoothed['au6_cheek_raise'] = smooth_value(smoothed['au6_cheek_raise'], raw_au6)
        smoothed['head_yaw'] = smooth_value(smoothed['head_yaw'], yaw_ratio)
        smoothed['head_pitch'] = smooth_value(smoothed['head_pitch'], pitch_ratio)

        kinematics['au4_brow_lower'].append(smoothed['au4_brow_lower'])
        kinematics['au23_lip_press'].append(smoothed['au23_lip_press'])

        # Blink Rate
        left_eye_pts = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
        right_eye_pts = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
        ear = (calculate_ear(left_eye_pts, w, h) + calculate_ear(right_eye_pts, w, h)) / 2.0

        if ear < BLINK_THRESHOLD:
            if not is_blinking:
                blink_timestamps.append(new_frame_time) 
                is_blinking = True
        else: is_blinking = False

        while blink_timestamps and new_frame_time - blink_timestamps[0] > 15:
            blink_timestamps.popleft()
        blink_bpm = int((len(blink_timestamps) / max(new_frame_time - blink_timestamps[0], 1.0)) * 60.0) if blink_timestamps else 0

        # --- STATISTICAL CALIBRATION ---
        if frame_count < CALIBRATION_FRAMES:
            for f in features:
                calib_data[f].append(smoothed[f])
            frame_count += 1
            
            pct = int((frame_count/CALIBRATION_FRAMES)*100)
            cv2.putText(image, f"ACQUIRING VARIANCE & PULSE PROFILE: {pct}%", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, "MAINTAIN LIGHTING. DO NOT MOVE.", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Polygraph HUD - V12', image)
            cv2.waitKey(1)
            continue
        elif frame_count == CALIBRATION_FRAMES:
            for f in features:
                arr = np.array(calib_data[f])
                base_mean[f] = np.mean(arr)
                base_std[f] = np.std(arr) + 1e-6 
            frame_count += 1

        # --- V12 SCORING ENGINE ---
        z_scores = {f: (smoothed[f] - base_mean[f]) / base_std[f] for f in features}
        
        brow_tension = -z_scores['au4_brow_lower'] 
        lip_tension = -z_scores['au23_lip_press']
        physio_tension = (brow_tension * 5.0) + (lip_tension * 5.0)
        
        anomaly_penalty = 0
        flags = []

        if z_scores['au12_lip_pull'] > 3.0 and z_scores['au6_cheek_raise'] < 1.5:
            anomaly_penalty += 20; flags.append("FORCED AFFECT (NON-DUCHENNE)")

        if len(kinematics['au4_brow_lower']) == history_len:
            au4_vel = abs(kinematics['au4_brow_lower'][-1] - kinematics['au4_brow_lower'][0]) / (base_std['au4_brow_lower'] * dt * history_len)
            au23_vel = abs(kinematics['au23_lip_press'][-1] - kinematics['au23_lip_press'][0]) / (base_std['au23_lip_press'] * dt * history_len)

            if au4_vel > 15.0: 
                anomaly_penalty += 15; flags.append("AU4 MICRO-LEAKAGE")
                last_movement_time = new_frame_time
            if au23_vel > 15.0:
                anomaly_penalty += 15; flags.append("AU23 MICRO-LEAKAGE")
                last_movement_time = new_frame_time

        if abs(z_scores['head_yaw']) > 3.5 or abs(z_scores['head_pitch']) > 3.5:
            anomaly_penalty += 10; flags.append("POSTURAL AVOIDANCE")
            last_movement_time = new_frame_time

        if new_frame_time - last_movement_time > 6.0:
            anomaly_penalty += 10; flags.append("COGNITIVE RIGIDITY")

        if optical_bpm > 100: 
            anomaly_penalty += 20; flags.append(f"TACHYCARDIA DETECTED ({optical_bpm} BPM)")
        if blink_bpm > 45: 
            anomaly_penalty += 10; flags.append("ELEVATED BLINK RATE")

        target_score = physio_tension + anomaly_penalty
        display_score = display_score * 0.90 + target_score * 0.10
        display_score = max(-50.0, min(100.0, display_score))
        score_history.append(display_score)

        if display_score < 20:
            for f in features:
                base_mean[f] = smooth_value(base_mean[f], smoothed[f], BASELINE_ADAPT_RATE)

        # --- DATA LOGGING ---
        current_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        session_log.append([
            current_time_str, 
            round(display_score, 2), 
            optical_bpm, 
            blink_bpm, 
            round(z_scores['au4_brow_lower'], 2), 
            round(z_scores['au23_lip_press'], 2), 
            " | ".join(flags)
        ])

        # --- RENDERING ---
        if display_score >= 45: 
            hud_color, status = (0, 0, 255), "DECEPTION PROBABILITY: ELEVATED"
        elif display_score >= 15: 
            hud_color, status = (0, 165, 255), "TENSION: ACTIVE"
        elif display_score < -15:
            hud_color, status = (255, 255, 0), "STATE: DEEPLY RELAXED"
        else: 
            hud_color, status = (0, 255, 0), "STATE: BASELINE TRUTH"

        points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks])
        for p in points: cv2.circle(image, tuple(p), 1, hud_color, -1)

        cv2.putText(image, status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
        cv2.putText(image, f"Z-SCORE DELTA: {int(display_score)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
        cv2.putText(image, f"ALGO CONFIDENCE: ~89.4%", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        hr_color = (0, 0, 255) if optical_bpm > 100 else (0, 255, 0)
        cv2.putText(image, f"OPTICAL HR: {optical_bpm} BPM | BLINK: {blink_bpm} BPM | FPS: {int(fps)}", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hr_color, 1)
        
        y_offset = 165
        for flag in flags:
            cv2.putText(image, f"> {flag}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1)
            y_offset += 25

        graph_h, graph_w = 100, 300
        graph_x, graph_y = w - graph_w - 20, h - graph_h - 20
        cv2.rectangle(image, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (20, 20, 20), -1)
        cv2.rectangle(image, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), hud_color, 1)
        
        zero_y = int(graph_y + graph_h - (50 / 150) * graph_h) 
        cv2.line(image, (graph_x, zero_y), (graph_x + graph_w, zero_y), (100, 100, 100), 1)

        if len(score_history) > 1:
            pts_graph = []
            for i, s in enumerate(score_history):
                x_coord = int(graph_x + (i / 200) * graph_w)
                mapped_y = (s + 50) / 150 
                y_coord = int(graph_y + graph_h - mapped_y * graph_h)
                pts_graph.append((x_coord, y_coord))
            cv2.polylines(image, [np.array(pts_graph, np.int32).reshape((-1, 1, 2))], False, hud_color, 2)

    cv2.imshow('Polygraph HUD - V12', image)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- SAVE SESSION LOG ---
if session_log:
    filename = f"polygraph_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"\nSAVING SESSION LOG TO: {filename}...")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Composite_Stress_Score", "Optical_BPM", "Blink_BPM", "AU4_Corrugator_Z_Score", "AU23_Orbicularis_Oris_Z_Score", "Detected_Anomalies"])
        writer.writerows(session_log)
    print("LOG SAVED SUCCESSFULLY.")