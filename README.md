# Piano Hand Detector (Python + OpenCV + MediaPipe)

This is a simple desktop app that detects hands and faces from your webcam and draws landmarks and semi-transparent bounding boxes. It plays a beep when a hand reliably appears and shows counts for hands and faces. You can tune detection thresholds and smoothing at runtime using the built-in Settings panel.

Key features
- Hand detection (MediaPipe Hands) with improved finger/thumb detection (angle-based thumb heuristic).
- Face detection (MediaPipe Face Detection) with per-face tracking and smoothed bounding boxes.
- Runtime Settings UI (OpenCV trackbars) to tune detection confidences, thumb angle threshold and face smoothing without restarting.
- Visual overlays and simple audio feedback when hands appear.

Requirements
- Windows with a webcam (the app includes Windows-specific code to bring the window forward).
- Python 3.8+ (3.8–3.11 recommended).
- Packages in `requirements.txt` (OpenCV, MediaPipe, numpy).

Installation (PowerShell)
```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

Run
```powershell
python .\app.py
```

Usage
- When launched you'll be asked for camera permission (simple tkinter prompt). Click Yes to continue.
- The camera window opens and a separate "Settings" window appears with sliders:
  - Hand Det % — MediaPipe Hands detection confidence (0–100%).
  - Hand Track % — MediaPipe Hands tracking confidence (0–100%).
  - Face Det % — MediaPipe Face Detection confidence (0–100%).
  - Thumb Angle — angle threshold in degrees used to decide if a thumb is extended (higher = stricter).
  - Face Smooth N — number of frames used for face smoothing (higher = smoother but slower to react).
- Change sliders at runtime — the app recreates detectors with the new thresholds automatically.
- Press `q` in the camera window to quit.

What to expect
- Hands: landmarks and hand bounding boxes are drawn. Open/Closed counts use a finger-count heuristic with an improved geometric thumb test.
- Faces: each face is assigned a small track ID (Face #N) and a smoothed bounding box to reduce jitter. The number of currently-detected faces is shown on-screen.

Troubleshooting
- If the camera doesn't open, close apps that may be using the camera and verify Windows privacy settings.
- If pip fails to install `mediapipe`, ensure you're using a supported Python version and have an up-to-date pip. Some Windows users may need Visual C++ build tools installed for certain wheels.

Notes & future improvements
- The thumb detection uses a simple geometric angle; for maximum accuracy you can further combine multiple joint angles or train a small classifier.
- Face tracking uses centroid matching + EMA for smoothing. For robust multi-face tracking you could add IoU matching and identity persistence across occlusions.

Privacy
- This app runs locally and does not transmit frames over the network.

License
- MIT-style (adjust as needed)
