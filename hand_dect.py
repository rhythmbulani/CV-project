import cv2
import mediapipe as mp
import pyautogui
import math
import util  # your util.py must have get_angle and get_distance

# -----------------------------
# Initialize Mediapipe Hands
# -----------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# -----------------------------
# Screen info and smoothing
# -----------------------------
screen_width, screen_height = pyautogui.size()
sensitivity = 2      # pointer speed
noise_threshold = 2    # dead zone in pixels
history_length = 5     # smoothing frames

# Track last hand and pointer positions
hand_start_x, hand_start_y = None, None
pointer_x, pointer_y = None, None
hand_history_x = []
hand_history_y = []

# Gesture state
gesture_active = False

# Thresholds for fingers side by side
dx_threshold = 50
dy_threshold = 40

# -----------------------------
# Utility functions
# -----------------------------
def move_mouse_relative(index_finger_tip):
    global hand_start_x, hand_start_y, pointer_x, pointer_y, hand_history_x, hand_history_y

    if index_finger_tip is None:
        return

    x = index_finger_tip.x
    y = index_finger_tip.y

    # Initialize starting positions
    if hand_start_x is None or hand_start_y is None:
        hand_start_x = x
        hand_start_y = y
        pointer_x, pointer_y = pyautogui.position()
        return

    # Add to history for smoothing
    hand_history_x.append(x)
    hand_history_y.append(y)
    x_smooth = sum(hand_history_x[-history_length:]) / min(len(hand_history_x), history_length)
    y_smooth = sum(hand_history_y[-history_length:]) / min(len(hand_history_y), history_length)

    dx = (x_smooth - hand_start_x) * screen_width
    dy = (y_smooth - hand_start_y) * screen_height

    # Dead zone
    if abs(dx) < noise_threshold:
        dx = 0
    if abs(dy) < noise_threshold:
        dy = 0

    # Apply sensitivity
    dx *= sensitivity
    dy *= sensitivity

    pointer_x = max(0, min(screen_width, pointer_x + int(dx)))
    pointer_y = max(0, min(screen_height, pointer_y + int(dy)))
    pyautogui.moveTo(pointer_x, pointer_y)

    hand_start_x = x_smooth
    hand_start_y = y_smooth

# -----------------------------
# Gesture detection
# -----------------------------
def detect_gesture(frame, hand_landmarks):
    global gesture_active

    if hand_landmarks is None:
        gesture_active = False
        return

    # Extract landmarks
    landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    if len(landmark_list) < 21:
        gesture_active = False
        return

    index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]

    # Fingers side by side check using horizontal and vertical distances
    dx = abs(index_tip.x - middle_tip.x) * screen_width
    dy = abs(index_tip.y - middle_tip.y) * screen_height

    if dx < dx_threshold*1.5 and dy < dy_threshold*1.5:
        gesture_active = True
    else:
        gesture_active = False

    # Move pointer if gesture active
    if gesture_active:
        move_mouse_relative(index_tip)
        cv2.putText(frame, "Moving Pointer", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Left click: index finger curled
    elif util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50:
        pyautogui.click(button='left')
        cv2.putText(frame, "Left Click", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Right click: middle finger curled
    elif util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50:
        pyautogui.click(button='right')
        cv2.putText(frame, "Right Click", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# -----------------------------
# Main loop
# -----------------------------
def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                detect_gesture(frame, hand_landmarks)
            else:
                # Reset gesture state if no hand detected
                global gesture_active
                gesture_active = False

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
