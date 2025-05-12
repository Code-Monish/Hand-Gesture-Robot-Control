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

    # Thumb (left to right or right to left)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers: 1 if tip is above pip joint
    for tip in tips_ids[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def detect_gesture(fingers, lm):
    # Get y-coordinates for thumb tip and MCP joint
    thumb_tip_y = lm.landmark[4].y
    thumb_mcp_y = lm.landmark[2].y

    index_tip_y = lm.landmark[8].y
    index_mcp_y = lm.landmark[5].y

    # Gestures
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    elif fingers == [1, 0, 0, 0, 0] and thumb_tip_y < thumb_mcp_y:
        return "Thumbs Up"
    elif fingers == [1, 0, 0, 0, 0] and thumb_tip_y > thumb_mcp_y:
        return "Thumbs Down"
    elif fingers == [0, 1, 0, 0, 0] and index_tip_y < index_mcp_y:
        return "Index Up"
    elif fingers == [0, 1, 0, 0, 0] and index_tip_y > index_mcp_y:
        return "Index Down"
    else:
        return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = get_finger_states(hand_landmarks)
            gesture = detect_gesture(fingers, hand_landmarks)

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
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
