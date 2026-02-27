
# V12 Biometric Polygraph Matrix 👁️📊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An advanced, non-contact biometric analysis tool that utilizes 3D facial kinematics, micro-expression tracking, and remote Photoplethysmography (rPPG) to assess cognitive load, physiological tension, and autonomic arousal in real-time.



## 🧠 Core Architecture

The V12 Engine establishes a dynamic resting baseline for a subject and calculates live Z-Scores to detect micro-anomalies. 

* **3D Facial Kinematics:** Tracks specific Facial Action Units (FAUs) including AU4 (Corrugator / Brow Lowerer), AU23 (Orbicularis Oris / Lip Presser), and AU12/AU6 (Duchenne Smile verification).
* **Remote Photoplethysmography (rPPG):** Optically extracts the subject's heart rate (BPM) by analyzing micro-fluctuations in light absorption on the forehead's skin tissue.
* **Autonomic Arousal Tracking:** Measures involuntary responses such as blink rate and postural avoidance (head yaw/pitch variance).
* **Statistical Z-Score Engine:** Standardizes live muscle contractions against the subject's personal, unique resting variance to prevent biased profiling.

## 🛠️ Installation & Prerequisites

1. Clone the repository:
   ```bash
   git clone [https://github.com/chaz-ux/v12-biometric-matrix.git](https://github.com/chaz-ux/v12-biometric-matrix.git)
   cd v12-biometric-matrix

```

2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. Ensure you have the MediaPipe Face Landmarker model downloaded (`face_landmarker.task`) and placed in the root directory.

## 🚀 Usage

Run the main script to initialize the camera and HUD:

```bash
python main.py

```

### Calibration Phase

For the first 200 frames, the system will instruct the subject to remain still. During this phase, it builds a statistical baseline of their resting facial micro-movements and optical heart rate. **Do not speak or move erratically during this phase.**

### Live Analysis

Once calibrated, the Polygraph HUD will display:

* Real-time Deception Probability / Stress Delta
* Live Optical HR and Blink Rate
* Triggered Anomaly Flags (e.g., "AU4 MICRO-LEAKAGE", "TACHYCARDIA DETECTED")

### 🛑 Ending the Session & Logging

To gracefully terminate the session and export the data, ensure the video window is in focus and press the **`q`** key.

The system will automatically generate a timestamped `.csv` file in the root directory containing frame-by-frame statistical breakdowns of the session.

## ⚠️ Disclaimer

*This software is intended for educational, research, and experimental purposes only. Micro-expression analysis and rPPG are not foolproof indicators of deception. Variables such as lighting, camera framerate, and baseline subject anxiety can impact the Z-score matrix. This tool should not be used for legal, medical, or official interrogative purposes.*

