import cv2
import mediapipe as mp
import numpy as np

class GestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    

    def classify_gesture(self, landmarks, frame_width, frame_height):
        wrist = landmarks[0]
        index_tip = landmarks[8]
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]

        # Gesture: Index Up
        if index_tip[1] < wrist[1]:
            return "Index Up"

        # Gesture: Index Down
        elif index_tip[1] > wrist[1]:
            return "Index Down"

        # Gesture: Thumb Right
        elif thumb_tip[0] > wrist[0]:
            return "Thumb Right"
    
        # Gesture: Thumb Left
        elif thumb_tip[0] < wrist[0]:
            return "Thumb Left"

        # Gesture: Open Palm
        elif abs(thumb_tip[0] - pinky_tip[0]) > 0.2 * frame_width:
            return "Open Palm"

        # Gesture: Fist
        elif abs(thumb_tip[0] - pinky_tip[0]) < 0.1 * frame_width:
            return "Fist"

        return None

    def detect_gesture(self, frame):
        """
        Detect gestures from a video frame.
        :param frame: Input video frame.
        :return: Detected gesture as a string.
        """
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract landmarks as a list of (x, y, z) tuples
                landmarks = [(lm.x * frame_width, lm.y * frame_height, lm.z) for lm in hand_landmarks.landmark]
                # Classify the gesture
                return self.classify_gesture(landmarks, frame_width, frame_height)
        return None


# Code below will only run if this script is executed directly
if __name__ == "__main__":
    # Start capturing live camera feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Press 'q' to quit.")

    gesture_recognizer = GestureRecognizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Detect gesture
        gesture = gesture_recognizer.detect_gesture(frame)

        if gesture:
            # Display the gesture on the frame
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Hand Gesture Recognition", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()