# Face Detector (Python + OpenCV)

Simple desktop camera face detector. When run, the app asks for camera permission (a simple GUI prompt). After you allow access the camera opens and the app draws a green border around detected faces with the label "Face".

Requirements
- Windows with a webcam
- Python 3.8+
- Packages in `requirements.txt`

Setup (PowerShell)
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run
```powershell
python app.py
```

Usage
- The app will show a permission prompt. Click Yes to allow the app to access your camera.
- A window named "Camera - Face Detector" will open. Faces will have a green rectangle and a label "Face".
- Press `q` to quit the camera window.

Notes
- The OS (Windows) may also ask for camera permission on first access; grant that too.
- If the camera is in use by another application, the app will fail to open the camera.
