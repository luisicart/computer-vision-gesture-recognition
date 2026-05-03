from pathlib import Path
import requests
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = Path("models")
MP_MODEL_PATH = MODEL_DIR / "gesture_recognizer.task"
CUSTOM_MODEL_PATH = MODEL_DIR / "custom_gesture_model.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"


def download_model():
    """Download the MediaPipe Gesture Recognizer model if not present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MP_MODEL_PATH.exists():
        print(f"Model already exists at: {MP_MODEL_PATH}")
        return MP_MODEL_PATH

    print("Downloading model...")

    try:
        response = requests.get(MODEL_URL, timeout=30)
        response.raise_for_status()

        with open(MP_MODEL_PATH, "wb") as f:
            f.write(response.content)

        print(f"Download complete: {MP_MODEL_PATH}")
        return MP_MODEL_PATH

    except requests.RequestException as e:
        print(f"Download failed: {e}")
        return None
    
def main():

    download_model()

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    print("--- Loading Custom Models ---")
    clf = joblib.load(CUSTOM_MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MP_MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    print("\nInicializing CUSTOM model... Press 'q' to quit.")

    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            
            recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            if recognition_result.hand_landmarks:
                for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
   
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    hand_label = recognition_result.handedness[i][0].category_name
                    handedness_val = 0 if hand_label == 'Left' else 1
                    
                    landmarks_array = [handedness_val]
                    for lm in hand_landmarks:
                        landmarks_array.extend([lm.x, lm.y, lm.z])
                    
                    columns = ['handedness'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
                    features = pd.DataFrame([landmarks_array], columns=columns)
                    
                    prediction_idx = clf.predict(features)[0]
                    prediction_prob = np.max(clf.predict_proba(features))
                    gesture_name = label_encoder.inverse_transform([prediction_idx])[0]

                    hand_label = recognition_result.handedness[i][0].category_name
                    
                    display_hand_label = 'Right' if hand_label == 'Left' else 'Left'
                    color = (0, 255, 0) 
                    
                    display_text = f"Custom {display_hand_label}: {gesture_name} ({prediction_prob:.2f})"
                    cv2.putText(frame, display_text, (20, 50 + (i * 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Custom Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()