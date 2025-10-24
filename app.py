import sys
import cv2
import tkinter as tk
from tkinter import messagebox
import winsound
import mediapipe as mp
import numpy as np


def ask_camera_permission():
    root = tk.Tk()
    root.withdraw()  # hide main window
    answer = messagebox.askyesno("Camera permission", "This app wants to access your camera. Allow?")
    root.destroy()
    return answer


def draw_hand_landmarks(frame, hand_landmarks, confidence=None):
    if not hand_landmarks:
        return
    
    # Convert landmarks to numpy array for easier processing
    landmarks = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in hand_landmarks.landmark])
    
    # Get bounding box
    x_min, y_min = np.min(landmarks, axis=0).astype(int)
    x_max, y_max = np.max(landmarks, axis=0).astype(int)
    
    # Draw semi-transparent bounding box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), -1)
    
    # Apply transparency
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw hand landmarks and connections
    mp_hands = mp.solutions.hands
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_point = tuple(map(int, landmarks[start_idx]))
        end_point = tuple(map(int, landmarks[end_idx]))
        
        cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
    
    for landmark in landmarks:
        point = tuple(map(int, landmark))
        cv2.circle(frame, point, 4, (0, 0, 255), -1)
    
    # Draw label with confidence
    label = f"Hand {confidence:.0f}%" if confidence else "Hand"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw label background (more opaque)
    label_overlay = frame.copy()
    cv2.rectangle(label_overlay, (x_min, y_min - text_h - 8), (x_min + text_w + 8, y_min), (0, 255, 0), -1)
    cv2.addWeighted(label_overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw white text
    cv2.putText(frame, label, (x_min + 4, y_min - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def run_hand_detector(camera_index=0):
    # Use DirectShow backend on Windows to avoid some warnings
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    except TypeError:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera. Make sure it's connected and not used by another app.")
        return

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    window_name = "Camera - Hand Detector"
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        # Draw hand count and make beep sound if hands are detected
        hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        if hand_count > 0:
            winsound.Beep(1000, 100)  # Frequency: 1000Hz, Duration: 100ms
            
        count_text = f"Hands detected: {hand_count}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate confidence based on landmark positions
                confidence = 100 * min(
                    max(
                        np.mean([lm.z for lm in hand_landmarks.landmark]) + 0.5,
                        0
                    ),
                    1
                )
                
                draw_hand_landmarks(frame, hand_landmarks, confidence)

        cv2.imshow(window_name, frame)

        # Quit when 'q' is pressed or window closed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()


def main():
    allowed = ask_camera_permission()
    if not allowed:
        print("Camera permission not granted. Exiting.")
        sys.exit(0)

    print("Opening camera. Press 'q' to quit.")
    run_hand_detector(0)


if __name__ == "__main__":
    main()
