import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

def get_finger_states(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers: 1 if tip is above pip joint
    for tip in tips_ids[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def classify_gesture(hand_landmarks):
    """
    Classify gestures based on finger states and relative positions.
    """
    fingers = get_finger_states(hand_landmarks)

    # Gesture: Index Up
    if fingers == [0, 1, 0, 0, 0]:
        return "Index Up"

    # Gesture: Index Down
    elif fingers == [0, 0, 0, 0, 0]:
        return "Index Down"

    # Gesture: Thumb Right
    elif fingers == [1, 0, 0, 0, 0] and hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
        return "Thumb Right"

    # Gesture: Thumb Left
    elif fingers == [1, 0, 0, 0, 0] and hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        return "Thumb Left"

    # Gesture: Open Palm
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"

    # Gesture: Fist
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist"

    return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Classify gesture
            gesture = classify_gesture(hand_landmarks)

            # Display the detected gesture
            cv2.putText(frame, f'Gesture: {gesture}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Map gestures to actions (for integration with robot.py)
            if gesture == "Index Up":
                print("Move prismatic joint up")
            elif gesture == "Index Down":
                print("Move prismatic joint down")
            elif gesture == "Thumb Right":
                print("Rotate first revolute joint forward")
            elif gesture == "Thumb Left":
                print("Rotate first revolute joint backward")
            elif gesture == "Open Palm":
                print("Rotate second revolute joint forward")
            elif gesture == "Fist":
                print("Rotate second revolute joint backward")

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
