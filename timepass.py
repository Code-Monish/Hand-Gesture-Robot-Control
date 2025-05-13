import os
import cv2
import numpy as np
import time
import threading
import pybullet as p
import pybullet_data
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

# Load ground plane and robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("SCARA.urdf", basePosition=[0, 0, 0.1], useFixedBase=True)

# Create directories for saving images
output_dir = "gesture_database"
os.makedirs(output_dir, exist_ok=True)

# Initialize variables for gesture tracking
gesture_start_time = None
last_detected_gesture = None
save_interval = 5  # Save image after 5 seconds of consistent gesture

# Joint indices and step size
revolute_joint_1 = 0  # First revolute joint
revolute_joint_2 = 1  # Second revolute joint
prismatic_joint = 2   # Prismatic joint
step_size = 0.1       # Step size for joint adjustments

# Shared variables for threading
frame_to_save = None
gesture_to_save = None
save_lock = threading.Lock()

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
    return np.uint8(img_back)

def save_gesture_image():
    """
    Background thread function to save processed images.
    """
    global frame_to_save, gesture_to_save

    while True:
        if frame_to_save is not None and gesture_to_save is not None:
            with save_lock:
                frame = frame_to_save
                gesture = gesture_to_save
                frame_to_save = None
                gesture_to_save = None

            # Create gesture-specific directory
            gesture_dir = os.path.join(output_dir, gesture)
            os.makedirs(gesture_dir, exist_ok=True)

            # Apply high-pass Fourier transform
            processed_image = apply_high_pass_fourier(frame)

            # Save the image
            timestamp = int(time.time())
            filename = os.path.join(gesture_dir, f"{gesture}_{timestamp}.png")
            cv2.imwrite(filename, processed_image)
            print(f"Saved image: {filename}")

        time.sleep(0.1)  # Avoid busy-waiting

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
    elif fingers == [0, 1, 0, 0, 0] and index_tip_y > index_mcp_y:
        return "Index Down"
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

# Start the background thread for saving images
threading.Thread(target=save_gesture_image, daemon=True).start()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
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
                last_positions[prismatic_joint] = move_joint(prismatic_joint, last_positions[prismatic_joint], step_size, -1.0, 1.0)
            elif gesture == "Index Down":
                last_positions[prismatic_joint] = move_joint(prismatic_joint, last_positions[prismatic_joint], -step_size, -1.0, 1.0)
            elif gesture == "Thumb Up":
                last_positions[revolute_joint_1] = move_joint(revolute_joint_1, last_positions[revolute_joint_1], step_size, -3.14, 3.14)
            elif gesture == "Thumb Down":
                last_positions[revolute_joint_1] = move_joint(revolute_joint_1, last_positions[revolute_joint_1], -step_size, -3.14, 3.14)
            elif gesture == "Open Palm":
                last_positions[revolute_joint_2] = move_joint(revolute_joint_2, last_positions[revolute_joint_2], step_size, -3.14, 3.14)
            elif gesture == "Fist":
                last_positions[revolute_joint_2] = move_joint(revolute_joint_2, last_positions[revolute_joint_2], -step_size, -3.14, 3.14)

            # Save the frame and gesture for background processing
            if gesture == last_detected_gesture:
                if gesture_start_time and time.time() - gesture_start_time >= save_interval:
                    with save_lock:
                        frame_to_save = frame.copy()
                        gesture_to_save = gesture
                    gesture_start_time = None  # Reset the timer
            else:
                last_detected_gesture = gesture
                gesture_start_time = time.time()

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