import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 
import os


model_path = "gesture_recognizer.task"

if not os.path.exists(model_path):
    print(f"Error: Model path does not exist: {model_path}")
else:
    print(f"Model found at: {model_path}")


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult 
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result = GestureRecognizerResult, output_image = mp.Image, timestamp_ms =int):

    try:
        if result.gestures:
            for gesture in result.gestures:
                print(f"Detected Gesture: {gesture}")  
                
                if isinstance(gesture, list):
                    for category in gesture:
                        print(f"Gesture Category: {category.display_name}, Score: {category.score}")
                else:
                    print(f"Gesture does not have 'name' attribute. Structure: {gesture}")
        
        else:
            print("No gesture detected.")
        
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                print("Hand landmarks detected.")
                mp_drawing.draw_landmarks(output_image, landmarks, mp_hands.HAND_CONNECTIONS)

                # for lm in landmarks.landmark:
                    # landmark_list.append((lm.x, lm.y))
        
    except Exception as e:
        print(f"Error in result callback: {e}")

        

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)


with GestureRecognizer.create_from_options(options) as recognizer:
   
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 


    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame.")
            break
       
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(time.time() * 1000)  
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        cv2.imshow("Screen", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

    