import sys
import cv2
import tkinter as tk
from tkinter import messagebox


def ask_camera_permission():
    root = tk.Tk()
    root.withdraw()  # hide main window
    answer = messagebox.askyesno("Camera permission", "This app wants to access your camera. Allow?")
    root.destroy()
    return answer


def draw_translucent_overlay(frame, x, y, w, h, text, confidence=None):
    overlay = frame.copy()
    
    # Draw semi-transparent rectangle
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)
    
    # Apply transparency
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw label with confidence
    label = f"Face {confidence:.0f}%" if confidence else "Face"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw label background (more opaque)
    label_overlay = frame.copy()
    cv2.rectangle(label_overlay, (x, y - text_h - 8), (x + text_w + 8, y), (0, 255, 0), -1)
    cv2.addWeighted(label_overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw white text
    cv2.putText(frame, label, (x + 4, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def run_face_detector(camera_index=0):
    # Use DirectShow backend on Windows to avoid some warnings
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    except TypeError:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera. Make sure it's connected and not used by another app.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    window_name = "Camera - Face Detector"
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Improved detection parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,  # Increased for better accuracy
            minSize=(40, 40),  # Slightly larger minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw face count
        face_count = len(faces)
        count_text = f"Faces detected: {face_count}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        for (x, y, w, h) in faces:
            # Calculate confidence based on face size and position
            face_size_score = min((w * h) / (frame.shape[0] * frame.shape[1]) * 1000, 100)
            center_score = (1 - abs(x + w/2 - frame.shape[1]/2) / frame.shape[1]) * 100
            confidence = (face_size_score + center_score) / 2
            
            draw_translucent_overlay(frame, x, y, w, h, "Face", confidence)

        cv2.imshow(window_name, frame)

        # Quit when 'q' is pressed or window closed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    allowed = ask_camera_permission()
    if not allowed:
        print("Camera permission not granted. Exiting.")
        sys.exit(0)

    print("Opening camera. Press 'q' to quit.")
    run_face_detector(0)


if __name__ == "__main__":
    main()
