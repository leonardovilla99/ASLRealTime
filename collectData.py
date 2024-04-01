import cv2
import numpy as np
import os
import mediapipe as mp

# Variables
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['thanks'])
sequence_number = sequence_length = 30

# Create folder for actions
for action in actions:
    for sequence in range(sequence_number):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Holistic and drawing
holistic = mp.solutions.holistic
drawing = mp.solutions.drawing_utils

# Mediapipe detection function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Drawing landmarks on image function
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

# Extract keypoint function
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# Collect datapoint and store
cap = cv2.VideoCapture(2)

# Set mediapipe model
with holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(sequence_number):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_landmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames number {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('ASL Train', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames number {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('ASL Train', image)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                path_to_save = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(path_to_save, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
