import pybullet as p
import pybullet_data
import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For plane.urdf or others
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

# Load ground plane and robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("SCARA.urdf", basePosition=[0, 0, 0.1], useFixedBase=True)

# Get joint information
num_joints = p.getNumJoints(robot_id)
print(f"\nRobot has {num_joints} joints.\n")
joint_ids = []
last_positions = [0.0] * num_joints

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_ids.append(i)
    print(f"Joint {i}: name = {joint_info[1].decode()}, type = {joint_info[2]}")

# Start capturing live camera feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

# Joint indices and step size
revolute_joint_1 = 0  # First revolute joint
revolute_joint_2 = 1  # Second revolute joint
prismatic_joint = 2   # Prismatic joint
step_size = 0.1       # Step size for joint adjustments

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
        return "Thumb Up"
    elif fingers == [1, 0, 0, 0, 0] and thumb_tip_y > thumb_mcp_y:
        return "Thumb Down"
    elif fingers == [0, 1, 0, 0, 0] and index_tip_y < index_mcp_y:
        return "Index Up"
    elif fingers == [0, 1, 1, 0, 0] and index_tip_y < index_mcp_y:
        return "Peace"
    else:
        return "Unknown"

def move_joint(joint_id, current_value, delta, min_value, max_value):
    """
    Move a joint by a specified delta, clamping the value within a range.
    """
    new_value = current_value + delta
    new_value = max(min(new_value, max_value), min_value)
    p.setJointMotorControl2(robot_id, joint_id, p.POSITION_CONTROL, targetPosition=new_value)
    return new_value

while True:
    # Read the camera frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = get_finger_states(hand_landmarks)
            gesture = detect_gesture(fingers, hand_landmarks)

            # Display the gesture on the frame
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Detected Gesture: {gesture}")

            # Map gestures to robot actions
            if gesture == "Index Up":
                # Move the prismatic joint up
                current_value = last_positions[prismatic_joint]
                new_value = current_value + step_size
                new_value = max(min(new_value, 1.0), -1.0)  # Ensure within valid range
                p.setJointMotorControl2(robot_id, joint_ids[prismatic_joint], p.POSITION_CONTROL, targetPosition=new_value)
                last_positions[prismatic_joint] = new_value
            elif gesture == "Peace":
                # Move the prismatic joint down
                current_value = last_positions[prismatic_joint]
                new_value = current_value - step_size
                new_value = max(min(new_value, 1.0), -1.0)  # Ensure within valid range
                p.setJointMotorControl2(robot_id, joint_ids[prismatic_joint], p.POSITION_CONTROL, targetPosition=new_value)
                last_positions[prismatic_joint] = new_value
            elif gesture == "Thumb Up":
                # Rotate the first revolute joint forward
                current_value = last_positions[revolute_joint_1]
                new_value = current_value + step_size
                new_value = min(new_value, 3.14)  # Ensure within valid range
                p.setJointMotorControl2(robot_id, joint_ids[revolute_joint_1], p.POSITION_CONTROL, targetPosition=new_value)
                last_positions[revolute_joint_1] = new_value
            elif gesture == "Thumb Down":
                # Rotate the first revolute joint backward
                current_value = last_positions[revolute_joint_1]
                new_value = current_value - step_size
                new_value = max(new_value, -3.14)  # Ensure within valid range
                p.setJointMotorControl2(robot_id, joint_ids[revolute_joint_1], p.POSITION_CONTROL, targetPosition=new_value)
                last_positions[revolute_joint_1] = new_value
            elif gesture == "Open Palm":
                # Rotate the second revolute joint forward
                current_value = last_positions[revolute_joint_2]
                new_value = current_value + step_size
                new_value = min(new_value, 3.14)  # Ensure within valid range
                p.setJointMotorControl2(robot_id, joint_ids[revolute_joint_2], p.POSITION_CONTROL, targetPosition=new_value)
                last_positions[revolute_joint_2] = new_value
            elif gesture == "Fist":
                # Rotate the second revolute joint backward
                current_value = last_positions[revolute_joint_2]
                new_value = current_value - step_size
                new_value = max(new_value, -3.14)  # Ensure within valid range
                p.setJointMotorControl2(robot_id, joint_ids[revolute_joint_2], p.POSITION_CONTROL, targetPosition=new_value)
                last_positions[revolute_joint_2] = new_value
    else:
        print("No hand detected")
        
    # Step the simulation
    p.stepSimulation()

    # Show the camera feed
    cv2.imshow("Camera Input with Gestures", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()