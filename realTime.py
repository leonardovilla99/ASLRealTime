## Working model
# 3_newAction_v2 ('nice','meet','you')
# 6_newAction_v1 ('hello','iloveyou','thanks','nice','meet','you')

import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Actions
actions = np.array(['hello','iloveyou','thanks','nice','meet','you'])
model_name = '6_newAction_v1'
# Path
DATA_PATH = os.path.join('MP_Data')
# Video variable
sequence_number = sequence_length = 30

# Holistic
holistic = mp.solutions.holistic
drawing = mp.solutions.drawing_utils

# Functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # Draw left hand connections
    drawing.draw_landmarks(image, results.left_hand_landmarks, holistic.HAND_CONNECTIONS,
                             drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    drawing.draw_landmarks(image, results.right_hand_landmarks, holistic.HAND_CONNECTIONS,
                             drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights(model_name + '.h5')

sequence = []
sentence = []
threshold = 0.9

cap = cv2.VideoCapture(2)
# Set mediapipe model
with holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            # Video add text
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
            if len(sentence) > 10:
                sentence = sentence[-10:]

        cv2.putText(image, ' '.join(sentence), (5,40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('ASL RealTime Detection', image)

        # Exit screen
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
