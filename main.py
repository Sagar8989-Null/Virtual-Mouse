import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from pynput.mouse import Button, Controller
import threading

# Setup
mouse = Controller()
screen_width, screen_height = pyautogui.size()
mpHands = mp.solutions.hands
scale = (2 / 1)  # (mouse/hand)
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
    max_num_hands=1
)

is_dragging = False
last_click_time = 0 
click_cooldown = 1  # in seconds

def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landmarks_list):
    if len(landmarks_list) < 2:
        return

    (x1, y1), (x2, y2) = landmarks_list[0], landmarks_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        frame_center_x = screen_width / 2
        x = int(index_finger_tip.x * screen_width * scale)
        y = int(index_finger_tip.y * screen_height * scale)
        mouse.position = (x, y)

def is_left_click(landmarks_list, thumb_index_dist):
    return (get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
            get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) > 90 and
            thumb_index_dist > 150)

def is_right_click(landmarks_list, thumb_index_dist):
    return (
        get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
        get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
        thumb_index_dist > 150
    )

def is_double_click(landmarks_list, thumb_index_dist):
    return (
        get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
        get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 90 and
        thumb_index_dist > 150
    )

def is_screenshot(landmarks_list, thumb_index_dist):
    return (
        get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
        get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
        thumb_index_dist < 50
    )

def handle_drag_and_drop(landmarks_list, processed):
    global is_dragging
    thumb_index_dist = get_distance([landmarks_list[4], landmarks_list[8]])
    pinky_dist = get_distance([landmarks_list[16], landmarks_list[4]])  
    middle_dist = get_distance([landmarks_list[12], landmarks_list[4]])  
    ring_dist = get_distance([landmarks_list[20], landmarks_list[4]]) 

    if thumb_index_dist < 30 and not is_dragging:
        if pinky_dist > 50 and middle_dist > 50 and ring_dist > 50:
            print("Pinch detected: Start dragging")
            mouse.press(Button.left)  
            is_dragging = True

    if thumb_index_dist > 80 and is_dragging:  
        print("Pinch released: Stop dragging")
        mouse.release(Button.left) 
        is_dragging = False

    if is_dragging:
        index_finger_tip = find_finger_tip(processed)
        move_mouse(index_finger_tip)

def detect_gestures(frame, landmarks_list, processed):
    global last_right_click_time

    if len(landmarks_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = get_distance([landmarks_list[4], landmarks_list[5]])

        handle_drag_and_drop(landmarks_list, processed)

        if thumb_index_dist < 50 and get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90:
            move_mouse(index_finger_tip)

        # LEFT CLICK
        elif is_left_click(landmarks_list, thumb_index_dist):
            current_time = time.time()
            if current_time - last_click_time >= click_cooldown:
                mouse.press(Button.left)
                mouse.release(Button.left)
                last_click_time = current_time  
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # RIGHT CLICK
        elif is_right_click(landmarks_list, thumb_index_dist):
            current_time = time.time()
            if current_time - last_click_time >= click_cooldown:
                mouse.press(Button.right)
                mouse.release(Button.right)
                last_click_time = current_time  
            cv2.putText(frame, "Right Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # DOUBLE CLICK
        elif is_double_click(landmarks_list, thumb_index_dist):
            mouse.click(Button.left, 2)
            cv2.putText(frame, "Double Click", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # SCREENSHOT
        elif is_screenshot(landmarks_list, thumb_index_dist):
            timestamp = int(time.time())
            screenshot_filename = f"screenshots/screenshot_{timestamp}.png"
            pyautogui.screenshot(screenshot_filename)
            cv2.putText(frame, "Screenshot Taken", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # This line Shows the distance between the thumbtip and indextip
        # print(thumb_index_dist)

def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time() # Process starting time

            frame = cv2.flip(frame, 1)
            frame_resized = cv2.resize(frame, (640, 480))
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmarks_list = []

            threads = []
            if processed.multi_hand_landmarks:
                for hand_landmarks in processed.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                    for lm in hand_landmarks.landmark:
                        landmarks_list.append((lm.x, lm.y))

                # Create threads for gesture detection
                thread = threading.Thread(target=detect_gestures, args=(frame, landmarks_list, processed))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            end_time = time.time() # Process ending time  
            
            # Enable this part for displaying Processing time
            """
            latency = end_time - start_time 
            print(f"Processing time per frame: {latency:.4f} seconds")
            print(f"{latency:.4f}")
            """

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
