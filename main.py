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


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult  # This is necessary
VisionRunningMode = mp.tasks.vision.RunningMode

'''1'''
# Define the result callback function
# def print_result(result: GestureRecognizerResultq, output_image: mp.Image, timestamp_ms: int):
#     # Check if the result contains any recognized gestures
#     if result.gestures:
#         for gesture in result.gestures:
#             print(f"Gesture: {gesture.name}, Confidence: {gesture.confidence}")
#     else:
#         print("No gesture detected.")

'''2'''
# def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
#     print('Gesture recognition result: {}'.format(result))

'''3'''
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures:
        for gesture in result.gestures:
            print(f"Gesture: {gesture.name}, Confidence: {gesture.confidence}")
    else:
        print("No gesture detected.")
        
    if result.hand_landmarks:
        print(f"Hand landmarks: {result.hand_landmarks}")
    if result.hand_world_landmarks:
        print(f"Hand world landmarks: {result.hand_world_landmarks}")

        

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

    