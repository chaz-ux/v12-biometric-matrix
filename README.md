
<div align="center">

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=36&pause=1000&color=00FFCC&center=true&vCenter=true&width=800&lines=V12+Biometric+Polygraph+Matrix+👁️📊;FACIAL+KINEMATICS;REMOTE+PHOTOPLETHSYSMOGRAPHY;AUTONOMIC+AROUSAL+TRACKING)](https://git.io/typing-svg)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV Vision](https://img.shields.io/badge/OpenCV-Vision-green.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tracking-orange.svg?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![Security](https://img.shields.io/badge/Architecture-V20_CQT-red.svg?style=for-the-badge)](https://github.com/chaz-ux/v20-biometric-matrix)

[Architecture](#-core-architecture) • [Engines](#-modular-signal-engines) • [Installation](#%EF%B8%8F-installation--prerequisites) • [Usage](#-usage)

</div>

An advanced, non-contact biometric analysis tool that utilizes 3D facial kinematics, micro-expression tracking, and remote Photoplethysmography (rPPG). 

**V20 Major Overhaul:** Moving away from continuous procedural scoring, the V20 Engine introduces a highly structured **Control Question Test (CQT)** framework. It isolates autonomic responses into dedicated OOP modules, comparing physiological reactions during test questions against controlled baseline statements to deliver objective, statistically backed verdicts.

---

## 🧠 Core Architecture & CQT Framework

The V20 Engine drastically reduces false positives by forcing the subject through structured interrogation phases: **Neutral ➔ Truth ➔ Test ➔ Control**. 

Instead of arbitrarily penalizing movement, the system calculates a biometric delta between how the subject reacts to custom *Test* questions versus established *Control* questions (universal guilt prompts).

<details>
<summary><b>⚙️ View V20 Sensitivity Matrix Parameters</b></summary>
<br>

The framework allows for dynamic sensitivity tuning based on the context of the session:
- **LOW:** `di: 2, inc: 1` — Casual / party mode. Flags more anomalies, higher false-positive rate.
- **MED:** `di: 3, inc: 2` — General purpose baseline.
- **HIGH:** `di: 4, inc: 3` — Serious investigation. Conservative thresholds requiring highly correlated physiological spikes to flag deception.
</details>

---

## 🔬 Modular Signal Engines

Under the hood, V20 breaks down biometric tracking into four isolated, asynchronous OOP engines.

```text
v20-biometric-matrix/
├── main.py                 # Core CQT Loop & UI
├── requirements.txt        # Dependencies
├── face_landmarker.task    # MediaPipe Model Weights
└── engines/                # Isolated OOP Modules
    ├── OFEngine.py         # Postural Kinematics
    ├── rPPGEngine.py       # Optical Heart Rate
    ├── BlinkEngine.py      # Autonomic Eye Tracking
    └── FACSEngine.py       # Facial Action Units

```

| Engine | Primary Function | Technical Metric |
| --- | --- | --- |
| **`FACSEngine`** | Tracks Facial Action Units (FAUs) | Calculates AU4 (Corrugator), AU23 (Orbicularis Oris), and AU12/AU6 (Duchenne Smile verification) tension ratios. |
| **`rPPGEngine`** | Remote Heart Rate (HR) | Optically extracts BPM via FFT by analyzing micro-fluctuations in light absorption on forehead skin tissue. |
| **`BlinkEngine`** | Autonomic Eye Tracking | Tracks blink consolidation and fluttering by calculating the Eye Aspect Ratio (EAR) over rolling timeframes. |
| **`OFEngine`** | Postural Kinematics | Uses Farneback Optical Flow to detect micro-shifts, rigidity, and postural avoidance. |

---

## 🛠️ Installation & Prerequisites

1. Clone the repository:
```bash
git clone [https://github.com/chaz-ux/v20-biometric-matrix.git](https://github.com/chaz-ux/v20-biometric-matrix.git)
cd v20-biometric-matrix

```


2. Install the required dependencies:
```bash
pip install -r requirements.txt

```



> [!IMPORTANT]
> Ensure you have the MediaPipe Face Landmarker model downloaded (`face_landmarker.task`) and placed in the root directory before running the system.

---

## 🚀 Usage

Run the main script to initialize the graphical HUD:

```bash
python main.py

```

### 1. Setup Phase

The system will boot into the **Setup UI**. Here, you configure the interrogation parameters:

* Type a subject name or scenario label.
* Input up to 8 custom Test Questions.
* Toggle the sensitivity preset (`CTRL + S`).
* Once the UI displays **SYSTEM READY**, press `SPACE` to initiate the session.

### 2. Live Interrogation Phase

The system will automatically guide the subject through the CQT structure:

* **Truth Questions:** (e.g., "Say your first name.") to establish baseline variance.
* **Test Questions:** The custom questions you inputted during setup.
* **Control Questions:** Broad moral questions designed to elicit a baseline stress response for comparison.

### 3. Reveal & Results Phase

Upon session completion, the system transitions to the **Multi-Subject Reveal Screen**. It calculates the aggregate votes from all four signal engines to render a final verdict per question:

* **DI:** Deception Indicated 🟥
* **INC:** Inconclusive 🟨
* **NDI:** No Deception Indicated 🟩

---

## ⌨️ System Controls

| Action | Keybinding | Phase |
| --- | --- | --- |
| **Switch Input Field** | `TAB` | Setup |
| **Cycle Sensitivity** | `CTRL + S` | Setup |
| **Add Question** | `ENTER` | Setup |
| **Start Session** | `SPACE` | Setup |
| **Force Quit** | `ESC` | Global |

---

> [!WARNING]
> **DISCLAIMER:** This software is intended for educational, research, and experimental purposes only. Micro-expression analysis and rPPG are not foolproof indicators of deception. Variables such as lighting, camera framerate, and baseline subject anxiety can impact the analysis matrix. This tool should not be used for legal, medical, or official interrogative purposes.

```

