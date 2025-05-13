import pybullet as p
import pybullet_data
import time
import cv2
import mediapipe as mp
import numpy as np
import threading
import os
import queue

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

database_directory = "gesture_database"
if not os.path.exists(database_directory):
    os.makedirs(database_directory)
    
gesture_start_time = None
last_detected_gesture = None
save_interval = 2  # Save image after 5 seconds of consistent gesture

# Shared variables for threading
frame_to_save = None
gesture_to_save = None
save_lock = threading.Lock()

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

# Joint indices and step size
revolute_joint_1 = 0  # First revolute joint
revolute_joint_2 = 1  # Second revolute joint
prismatic_joint = 2   # Prismatic joint
step_size = 0.1       # Step size for joint adjustments

def apply_high_pass_fourier(image):
    """
    Apply a high-pass Fourier transform to highlight edges in the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a high-pass mask
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30  # Radius of the low-frequency region to block
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0

    # Apply the mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the result for visualization
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)  # Return the processed image

image_save_queue = queue.Queue()

def save_gesture_image_from_queue():
    """
    Function to process and save images from the queue.
    """
    while not image_save_queue.empty():
        # Get the frame and gesture from the queue
        frame, gesture = image_save_queue.get()

        # Create gesture-specific directory
        gesture_dir = os.path.join(database_directory, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        # Apply high-pass Fourier transform
        processed_image = apply_high_pass_fourier(frame)

        # Save the image
        timestamp = int(time.time())
        filename = os.path.join(gesture_dir, f"{gesture}_{timestamp}.png")
        cv2.imwrite(filename, processed_image)
        print(f"Saved image: {filename}")

        # Mark the task as done
        image_save_queue.task_done()

# def save_gesture_image():
#     """
#     Background thread function to save processed images.
#     """
#     global frame_to_save, gesture_to_save

#     while True:
#         if frame_to_save is not None and gesture_to_save is not None:
#             with save_lock:
#                 frame = frame_to_save
#                 gesture = gesture_to_save
#                 frame_to_save = None
#                 gesture_to_save = None

#             print(f"Saving image for gesture: {gesture}")

#             # Create gesture-specific directory
#             gesture_dir = os.path.join(database_directory, gesture)
#             os.makedirs(gesture_dir, exist_ok=True)

#             # Apply high-pass Fourier transform
#             processed_image = apply_high_pass_fourier(frame)

#             # Save the image
#             timestamp = int(time.time())
#             filename = os.path.join(gesture_dir, f"{gesture}_{timestamp}.png")
#             cv2.imwrite(filename, processed_image)
#             print(f"Saved image: {filename}")

#         time.sleep(0.1)  # Avoid busy-waiting


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

# threading.Thread(target=save_gesture_image, daemon=True).start()

# Main loop
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

            # Gesture consistency logic
            if gesture == last_detected_gesture:
                print("Gesture is consistent")
                if gesture_start_time and time.time() - gesture_start_time >= save_interval:
                    print("Saving gesture image...")
                    # Add the frame and gesture to the queue
                    image_save_queue.put((frame.copy(), gesture))
                    gesture_start_time = None
                else:
                    print("Gesture consistency timer reset.")
            else:
                print(f"New gesture detected: {gesture}")
                last_detected_gesture = gesture
                gesture_start_time = time.time()
    else:
        print("No hand detected.")

    # Process the image save queue
    save_gesture_image_from_queue()

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