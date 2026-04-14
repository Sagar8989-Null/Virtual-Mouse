# 🖐️ Hand Gesture Mouse Controller

Control your mouse using hand gestures detected via your webcam — no physical mouse needed. Built with MediaPipe, OpenCV, and Python.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Gesture Reference](#gesture-reference)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)

---

## Overview

This project uses your webcam and hand landmark detection (via MediaPipe) to translate hand gestures into real-time mouse actions — including movement, left/right clicks, double-click, drag-and-drop, and screenshots.

---

## Features

- 🖱️ **Mouse Movement** — Move the cursor using your index finger
- 👆 **Left Click** — Bend index finger down
- 🖱️ **Right Click** — Bend both index and middle fingers
- ✌️ **Double Click** — Partial bend of index finger
- 📸 **Screenshot** — Pinch gesture with close thumb-index distance
- 🤏 **Drag and Drop** — Pinch to grab, release to drop

---

## Requirements

- Python 3.7+
- Webcam

### Python Dependencies

```
opencv-python
mediapipe
pyautogui
numpy
pynput
```

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/hand-gesture-mouse.git
cd hand-gesture-mouse
```

2. **Install dependencies**

```bash
pip install opencv-python mediapipe pyautogui numpy pynput
```

3. **Create screenshots directory**

```bash
mkdir screenshots
```

4. **Run the application**

```bash
python gesture_mouse.py
```

Press **`Q`** to quit.

---

## How It Works

1. The webcam captures a live video feed, which is flipped horizontally for a mirror view.
2. MediaPipe detects 21 hand landmarks per frame.
3. Angles between finger joints and distances between fingertips are calculated to classify gestures.
4. Recognized gestures are mapped to mouse actions via `pynput` and `pyautogui`.

### Landmark Indices Used

| Landmark | Index |
|----------|-------|
| Thumb Tip | 4 |
| Index Finger MCP | 5 |
| Index Finger PIP | 6 |
| Index Finger Tip | 8 |
| Middle Finger MCP | 9 |
| Middle Finger PIP | 10 |
| Middle Finger Tip | 12 |
| Ring Finger Tip | 16 |
| Pinky Tip | 20 |

---

## Gesture Reference

| Gesture | Action | Trigger Condition |
|---------|--------|-------------------|
| Open hand / index pointing up | Mouse move | Index angle > 90°, thumb-index dist < 50 |
| Index finger curled | Left Click | Index angle < 50°, middle angle > 90°, thumb dist > 150 |
| Index + Middle curled | Right Click | Both angles < 50°, thumb dist > 150 |
| Index half-curled | Double Click | Index angle < 50°, middle angle < 90°, thumb dist > 150 |
| Both curled + close thumb | Screenshot | Both angles < 50°, thumb-index dist < 50 |
| Tight pinch (index + thumb) | Start Drag | Thumb-index dist < 30 |
| Release pinch | Stop Drag | Thumb-index dist > 80 |

---

## Configuration

You can adjust the following constants in the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `scale` | `2.0` | Mouse-to-hand movement multiplier |
| `click_cooldown` | `1` second | Minimum time between clicks |
| `min_detection_confidence` | `0.9` | MediaPipe hand detection threshold |
| `min_tracking_confidence` | `0.9` | MediaPipe hand tracking threshold |
| `max_num_hands` | `1` | Maximum hands tracked simultaneously |

---

## Known Limitations

- Works best in well-lit environments with a plain background.
- High `scale` values may make cursor control feel jittery.
- Double-click currently lacks a cooldown — rapid triggers may occur.
- Screenshot saving requires a `screenshots/` folder to exist before running.
- Intended for single-hand use only.

---

## License

MIT License — feel free to use, modify, and distribute.
