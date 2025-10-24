import sys
import cv2
import tkinter as tk
from tkinter import messagebox
import winsound
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter
import ctypes


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


def count_extended_fingers(hand_landmarks):
    """Return an approximate number of extended fingers for a single hand.

    This is a simple heuristic using landmark positions. It works reasonably
    well for upright palms facing the camera but is not perfect for all
    rotations/angles. Returns an int between 0 and 5.
    """
    lm = hand_landmarks.landmark
    count = 0

    # For index, middle, ring, pinky: compare tip.y with pip.y (lower y is up in image coordinates)
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip_idx, pip_idx in finger_pairs:
        try:
            if lm[tip_idx].y < lm[pip_idx].y:
                count += 1
        except Exception:
            pass

    # Thumb heuristic: if thumb tip is sufficiently far from thumb MCP in x-direction,
    # treat it as extended. This is a rough rule that works for many poses.
    try:
        if abs(lm[4].x - lm[2].x) > 0.04:
            count += 1
    except Exception:
        pass

    return max(0, min(5, count))


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
    # Create window early so FindWindow can find it on Windows
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    brought_to_front = False

    last_sound_time = 0  # Track when the last sound was played
    sound_cooldown = 0.5  # Cooldown period in seconds
    previous_hand_count = 0  # Track the previous (stable) number of hands

    # Buffer for temporal smoothing of detected hand counts
    stable_buffer = deque(maxlen=5)  # keep last 5 frames
    stable_required = 4  # require at least 4 of 5 frames to agree
    
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

        # Get current raw hand count
        hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        current_time = time.time()

        # Append to smoothing buffer and compute stable count only when buffer mostly agrees
        stable_buffer.append(hand_count)
        stable_count = previous_hand_count
        if len(stable_buffer) == stable_buffer.maxlen:
            most_common, freq = Counter(stable_buffer).most_common(1)[0]
            if freq >= stable_required:
                stable_count = most_common

        # Only act when stable count actually changes and cooldown has passed
        if stable_count != previous_hand_count and (current_time - last_sound_time) >= sound_cooldown:
            if stable_count > 0:  # Only beep when a hand appears
                winsound.Beep(523, 150)  # Frequency: 523Hz (C5 note), Duration: 150ms
            last_sound_time = current_time
            previous_hand_count = stable_count

        count_text = f"Hands detected: {stable_count}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Count open/closed hands and draw landmarks
        open_hands = 0
        closed_hands = 0
        if results.multi_hand_landmarks:
            # If MediaPipe provides handedness, zip to keep them aligned (not required here)
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate confidence based on landmark positions
                confidence = 100 * min(
                    max(
                        np.mean([lm.z for lm in hand_landmarks.landmark]) + 0.5,
                        0
                    ),
                    1
                )

                # Determine number of extended fingers and classify open/closed
                fingers = count_extended_fingers(hand_landmarks)
                if fingers >= 4:
                    open_hands += 1
                elif fingers <= 1:
                    closed_hands += 1

                draw_hand_landmarks(frame, hand_landmarks, confidence)

        # Draw open/closed counts at the top-right
        status_label = f"Open: {open_hands}  Closed: {closed_hands}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(status_label, font, font_scale, thickness)
        x = frame.shape[1] - text_w - 12
        y = 30
        # background rectangle
        label_overlay = frame.copy()
        cv2.rectangle(label_overlay, (x - 6, y - text_h - 6), (x + text_w + 6, y + 6), (0, 0, 0), -1)
        cv2.addWeighted(label_overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, status_label, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        # If the user clicked the window's close (X) button, stop the loop
        # cv2.getWindowProperty returns <1 when the window is closed
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            # Some backends may not support getWindowProperty; ignore and continue
            pass

        # Try to bring the window to the foreground once (Windows only)
        if not brought_to_front:
            try:
                # FindWindowW takes (lpClassName, lpWindowName)
                hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
                if hwnd:
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                    brought_to_front = True
            except Exception:
                # If anything goes wrong, don't block — it's a best-effort behavior
                brought_to_front = True

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
