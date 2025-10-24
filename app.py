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


def draw_face_detection(frame, detection):
    """Draws a bounding box, keypoints and confidence for a single face detection.

    Uses MediaPipe detection data (relative coordinates) and draws onto the BGR frame.
    """
    if not detection:
        return

    h, w = frame.shape[:2]
    try:
        bbox = detection.location_data.relative_bounding_box
        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)
        x_max = x_min + box_w
        y_max = y_min + box_h

        # Draw semi-transparent blue box for face
        overlay = frame.copy()
        color = (255, 0, 0)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        alpha = 0.15
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw keypoints if available
        try:
            for kp in detection.location_data.relative_keypoints:
                cx = int(kp.x * w)
                cy = int(kp.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
        except Exception:
            pass

        # Draw confidence label
        score = 0.0
        try:
            score = detection.score[0] if detection.score else 0.0
        except Exception:
            score = 0.0

        label = f"Face {int(score * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_x = max(0, x_min)
        label_y = max(0, y_min - 6)

        # label background
        label_overlay = frame.copy()
        cv2.rectangle(label_overlay, (label_x, label_y - text_h - 6), (label_x + text_w + 8, label_y + 4), color, -1)
        cv2.addWeighted(label_overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, label, (label_x + 4, label_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    except Exception:
        # If any of the expected fields are missing, ignore drawing for this detection
        pass


def _angle_between(p1, p2, p3):
    """Return angle at p2 formed by p1-p2-p3 in degrees."""
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    v1 = a - b
    v2 = c - b
    # Compute angle
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def _bbox_from_detection(detection, frame_w, frame_h):
    """Return bbox (x_min,y_min,x_max,y_max) in absolute pixel coords from a MediaPipe detection."""
    try:
        bbox = detection.location_data.relative_bounding_box
        x_min = int(bbox.xmin * frame_w)
        y_min = int(bbox.ymin * frame_h)
        box_w = int(bbox.width * frame_w)
        box_h = int(bbox.height * frame_h)
        return (x_min, y_min, x_min + box_w, y_min + box_h)
    except Exception:
        return None


def _centroid(bbox):
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)



def count_extended_fingers(hand_landmarks, handedness_label=None, thumb_angle_threshold=150.0):
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

    # Thumb geometric heuristic: compute the angle at the thumb IP joint
    # between the vectors (thumb_mcp -> thumb_ip) and (thumb_tip -> thumb_ip).
    # If the angle is large (near 180 deg) the thumb is extended.
    try:
        # indexes: 2=thumb_mcp, 3=thumb_ip, 4=thumb_tip
        angle = _angle_between(lm[2], lm[3], lm[4])
        # angle is between 0..180 where values closer to 180 indicate straighter thumb
        if angle >= thumb_angle_threshold:
            count += 1
        else:
            # Fallback small-x heuristic for edge cases
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
        min_detection_confidence=0.75,
        min_tracking_confidence=0.7
    )

    # Initialize MediaPipe Face Detection (keeps hands behavior unchanged)
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

    window_name = "Camera - Hand Detector"
    # Create window early so FindWindow can find it on Windows
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    brought_to_front = False

    # Settings UI (OpenCV trackbars) for runtime tuning
    settings_name = "Settings"
    cv2.namedWindow(settings_name, cv2.WINDOW_NORMAL)
    # initial values (percent integers)
    init_hand_det = int(0.75 * 100)
    init_hand_track = int(0.7 * 100)
    init_face_det = int(0.6 * 100)
    init_thumb_angle = 150
    init_face_smooth = 5

    cv2.createTrackbar("Hand Det %", settings_name, init_hand_det, 100, lambda x: None)
    cv2.createTrackbar("Hand Track %", settings_name, init_hand_track, 100, lambda x: None)
    cv2.createTrackbar("Face Det %", settings_name, init_face_det, 100, lambda x: None)
    cv2.createTrackbar("Thumb Angle", settings_name, init_thumb_angle, 180, lambda x: None)
    cv2.createTrackbar("Face Smooth N", settings_name, init_face_smooth, 20, lambda x: None)

    # Keep current applied values so we can recreate models when changed
    current_hand_det = init_hand_det / 100.0
    current_hand_track = init_hand_track / 100.0
    current_face_det = init_face_det / 100.0
    thumb_angle_threshold = float(init_thumb_angle)
    face_smooth_n = max(1, init_face_smooth)

    # Storage for per-face tracking and smoothing
    tracked_faces = {}  # id -> {bbox: (x1,y1,x2,y2), last_seen: frame_idx, score: float}
    next_face_id = 1
    frame_idx = 0
    track_alpha = 0.6  # EMA factor for smoothing bbox updates
    track_max_age = 10  # frames before a track is removed

    last_sound_time = 0  # Track when the last sound was played
    sound_cooldown = 0.5  # Cooldown period in seconds
    previous_hand_count = 0  # Track the previous (stable) number of hands

    # Buffer for temporal smoothing of detected hand counts
    stable_buffer = deque(maxlen=5)  # keep last 5 frames
    stable_required = 4  # require at least 4 of 5 frames to agree

    # Simple face-count smoothing buffer to avoid flicker in face count display
    face_buffer = deque(maxlen=5)
    face_required = 3
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Read trackbar values and recreate models if they've changed
        try:
            new_hand_det = cv2.getTrackbarPos("Hand Det %", settings_name) / 100.0
            new_hand_track = cv2.getTrackbarPos("Hand Track %", settings_name) / 100.0
            new_face_det = cv2.getTrackbarPos("Face Det %", settings_name) / 100.0
            new_thumb_angle = float(cv2.getTrackbarPos("Thumb Angle", settings_name))
            new_face_smooth = max(1, cv2.getTrackbarPos("Face Smooth N", settings_name))
        except Exception:
            # If trackbar window doesn't exist for any reason, keep using current values
            new_hand_det = current_hand_det
            new_hand_track = current_hand_track
            new_face_det = current_face_det
            new_thumb_angle = thumb_angle_threshold
            new_face_smooth = face_smooth_n

        # Recreate models if parameters changed
        if (abs(new_hand_det - current_hand_det) > 1e-3) or (abs(new_hand_track - current_hand_track) > 1e-3):
            try:
                hands.close()
            except Exception:
                pass
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=max(0.01, new_hand_det),
                                    min_tracking_confidence=max(0.01, new_hand_track))
            current_hand_det = new_hand_det
            current_hand_track = new_hand_track

        if abs(new_face_det - current_face_det) > 1e-3:
            try:
                face_detection.close()
            except Exception:
                pass
            face_detection = mp_face_detection.FaceDetection(min_detection_confidence=max(0.01, new_face_det))
            current_face_det = new_face_det

        # Update thumb angle and smoothing N
        thumb_angle_threshold = new_thumb_angle
        face_smooth_n = new_face_smooth

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # Process the frame and detect faces
        face_results = face_detection.process(rgb_frame)

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
            # If MediaPipe provides handedness, zip to keep them aligned
            handedness_list = results.multi_handedness if getattr(results, 'multi_handedness', None) else []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_label = None
                try:
                    if i < len(handedness_list):
                        handedness_label = handedness_list[i].classification[0].label
                except Exception:
                    handedness_label = None

                # Calculate confidence based on landmark positions (z mean heuristic)
                try:
                    confidence = 100 * min(max(np.mean([lm.z for lm in hand_landmarks.landmark]) + 0.5, 0), 1)
                except Exception:
                    confidence = None

                # Determine number of extended fingers and classify open/closed
                fingers = count_extended_fingers(hand_landmarks, handedness_label, thumb_angle_threshold)
                if fingers >= 4:
                    open_hands += 1
                elif fingers <= 1:
                    closed_hands += 1

                draw_hand_landmarks(frame, hand_landmarks, confidence)

        # Per-face tracking + smoothing: convert detections to pixel bboxes and match to tracked faces
        frame_idx += 1
        detections_list = []
        if face_results and getattr(face_results, 'detections', None):
            for det in face_results.detections:
                bbox_px = _bbox_from_detection(det, frame.shape[1], frame.shape[0])
                if bbox_px is None:
                    continue
                # preserve score if available
                try:
                    score = det.score[0] if det.score else 0.0
                except Exception:
                    score = 0.0
                detections_list.append({"bbox": bbox_px, "score": score, "raw": det})

        # Matching: greedy nearest-centroid matching
        unmatched_tracks = set(tracked_faces.keys())
        assigned_tracks = set()
        match_threshold = max(frame.shape[0], frame.shape[1]) * 0.20  # px

        for det in detections_list:
            dbbox = det["bbox"]
            dcent = _centroid(dbbox)
            best_id = None
            best_dist = float('inf')
            for tid in list(unmatched_tracks):
                tcent = _centroid(tracked_faces[tid]["bbox"])
                dist = np.hypot(tcent[0] - dcent[0], tcent[1] - dcent[1])
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None and best_dist <= match_threshold:
                # update track
                old_bbox = tracked_faces[best_id]["bbox"]
                # EMA smoothing
                new_bbox = tuple([
                    int((1 - track_alpha) * old_bbox[i] + track_alpha * dbbox[i]) for i in range(4)
                ])
                tracked_faces[best_id]["bbox"] = new_bbox
                tracked_faces[best_id]["last_seen"] = frame_idx
                tracked_faces[best_id]["score"] = det["score"]
                assigned_tracks.add(best_id)
                unmatched_tracks.discard(best_id)
            else:
                # create new track
                tracked_faces[next_face_id] = {
                    "bbox": dbbox,
                    "last_seen": frame_idx,
                    "score": det["score"]
                }
                assigned_tracks.add(next_face_id)
                # reserve id increment
                next_face_id += 1

        # Age out old tracks
        to_delete = [tid for tid, t in tracked_faces.items() if (frame_idx - t["last_seen"]) > track_max_age]
        for tid in to_delete:
            try:
                del tracked_faces[tid]
            except KeyError:
                pass

        # Draw tracks (only those seen recently)
        current_seen = 0
        for tid, t in tracked_faces.items():
            if (frame_idx - t["last_seen"]) <= 1:
                current_seen += 1
            x1, y1, x2, y2 = t["bbox"]
            color = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Face #{tid} {int(t.get('score',0)*100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # Display the number of currently-seen faces
        cv2.putText(frame, f"Faces detected: {current_seen}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 128, 0), 2, cv2.LINE_AA)

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

        # Before showing the frame, check whether the user closed the window
        # in the previous iteration. If we detect the window no longer exists,
        # break before calling cv2.imshow() — otherwise imshow() will recreate
        # the window and the program will continue running (this is why the
        # close (X) button sometimes appears to not quit the app).
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            # Some backends may not support getWindowProperty; ignore and continue
            pass

        # Windows-specific check: if the window handle no longer exists
        # (user closed it using the window manager) then exit. We do this
        # before imshow so the window isn't recreated by imshow().
        try:
            hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
            if not hwnd:
                break
        except Exception:
            # If ctypes or FindWindowW fails for any reason, ignore — we already
            # attempted a generic check above.
            pass

        cv2.imshow(window_name, frame)

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
    try:
        face_detection.close()
    except Exception:
        pass
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
