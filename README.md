```markdown
# Piano Hand Detector (Python + OpenCV + MediaPipe)

This is a simple desktop app that detects hands from your webcam and draws landmarks and a bounding box. It beeps when a hand appears, and shows how many hands are open vs closed in the top-right of the camera window.

Features
- Detects up to two hands using MediaPipe Hands
- Draws landmarks and a semi-transparent bounding box
- Plays a sound when a new hand reliably appears (debounced)
- Shows counts for "Open" and "Closed" hands in the top-right
- Brings the camera window to the foreground on Windows and supports closing via the window close (X) button or pressing `q`

Requirements
- Windows with a webcam (features use Windows APIs to bring window forward)
- Python 3.8+
- Packages in `requirements.txt` (OpenCV, MediaPipe, numpy, etc.)

Installation (PowerShell)
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run
```powershell
python .\app.py
```

Controls
- Click "Yes" on the camera permission dialog when prompted.
- The app window will open (it should be brought to the foreground automatically on Windows).
- Press `q` in the camera window to quit, or click the window close (X) button.

What the overlays mean
- "Hands detected: N" (top-left) — the stable number of hands detected (uses a short temporal buffer to avoid flicker).
- "Open: X  Closed: Y" (top-right) — counts of hands detected as open (palm) vs closed (fist) using a simple finger-count heuristic.

Troubleshooting
- If the camera doesn't open, make sure no other app is using it and that Windows privacy settings allow camera access.
- If detection is noisy, try moving to better lighting or tweak parameters in `app.py` (buffer size, required stable frames, cooldown).

Tuning
- At the top of `app.py` you can find parameters to tune responsiveness vs stability:
	- `stable_buffer` size and `stable_required` (how many frames must agree)
	- `sound_cooldown` (minimum seconds between beeps)

Privacy
- This app runs locally and uses your webcam. It does not send frames over the network.

License
- MIT-style (adjust as needed)

```
