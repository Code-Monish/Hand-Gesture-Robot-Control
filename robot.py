import pybullet as p
import pybullet_data
import time
from gesture_recognition import GestureRecognizer  # Import the gesture recognition module
import cv2
import numpy as np

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For plane.urdf or others

# Set up physics
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)  # You can also use real-time if needed

# Load ground plane (optional)
p.loadURDF("plane.urdf")

# Load your URDF (ensure paths are resolved properly)
robot_id = p.loadURDF("SCARA.urdf", basePosition=[0, 0, 0.1], useFixedBase=True)

# Check joints
num_joints = p.getNumJoints(robot_id)
print(f"\nRobot has {num_joints} joints.\n")

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    print(f"Joint {i}: name = {joint_info[1].decode()}, type = {joint_info[2]}")

sliders = []
joint_ids = []
last_positions = [0.0] * num_joints

for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = "Link" + joint_info[1].decode('utf-8')  # Decode the joint name to a string
    sliders.append(p.addUserDebugParameter(joint_name, -3.14, 3.14, 0))  # Pass joint_name as a string
    joint_ids.append(i)
    
print(f"Joint IDs: {joint_ids}")
print(f"Sliders: {sliders}")

time.sleep(5)  # Allow time for the GUI to initialize
# Initialize GestureRecognizer
gesture_recognizer = GestureRecognizer()

# Start capturing live camera feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

# Initialize joint indices and step size
revolute_joint_1 = 0  # First revolute joint
revolute_joint_2 = 1  # Second revolute joint
prismatic_joint = 2   # Prismatic joint
step_size = 0.1       # Step size for joint adjustments

for joint_id in joint_ids:
    print(f"Rotating Joint {joint_id} 360 degrees...")
    for angle in np.linspace(0, 2 * np.pi, num=100):  # Incrementally rotate from 0 to 2π radians
        p.setJointMotorControl2(robot_id, joint_id, p.POSITION_CONTROL, targetPosition=angle)
        p.stepSimulation()
        time.sleep(0.01)  # Small delay for smooth simulation
print("Mock rotation complete!")


while True:
    # Read the camera frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Detect gesture
    gesture = gesture_recognizer.detect_gesture(frame)
    print(f"Detected Gesture: {gesture}")  # Debug print for detected gesture

    # Map gestures to actions
    if gesture == "Index_Up":  # Replace with the actual gesture name from the model
        # Move the prismatic joint up
        current_value = last_positions[prismatic_joint]
        new_value = current_value + step_size
        new_value = max(min(new_value, 1.0), -1.0)  # Ensure within valid range
        print(f"Prismatic Joint moved up to {new_value}")
        p.setJointMotorControl2(robot_id, joint_ids[prismatic_joint], p.POSITION_CONTROL, targetPosition=new_value)
        last_positions[prismatic_joint] = new_value  # Update last known position
    elif gesture == "Index_Down":  # Replace with the actual gesture name from the model
        # Move the prismatic joint down
        current_value = last_positions[prismatic_joint]
        new_value = current_value - step_size
        new_value = max(min(new_value, 1.0), -1.0)  # Ensure within valid range
        print(f"Prismatic Joint moved down to {new_value}")
        p.setJointMotorControl2(robot_id, joint_ids[prismatic_joint], p.POSITION_CONTROL, targetPosition=new_value)
        last_positions[prismatic_joint] = new_value  # Update last known position
    elif gesture == "Thumb_Right":  # Replace with the actual gesture name from the model
        # Rotate the first revolute joint forward
        current_value = last_positions[revolute_joint_1]
        new_value = current_value + step_size
        new_value = min(new_value, 3.14)  # Ensure within valid range
        print(f"Revolute Joint 1 rotated forward to {new_value}")
        p.setJointMotorControl2(robot_id, joint_ids[revolute_joint_1], p.POSITION_CONTROL, targetPosition=new_value)
        last_positions[revolute_joint_1] = new_value  # Update last known position
    elif gesture == "Thumb_Left":  # Replace with the actual gesture name from the model
        # Rotate the first revolute joint backward
        current_value = last_positions[revolute_joint_1]
        new_value = current_value - step_size
        new_value = max(new_value, -3.14)  # Ensure within valid range
        print(f"Revolute Joint 1 rotated backward to {new_value}")
        p.setJointMotorControl2(robot_id, joint_ids[revolute_joint_1], p.POSITION_CONTROL, targetPosition=new_value)
        last_positions[revolute_joint_1] = new_value  # Update last known position
    elif gesture == "Open_Palm":  # Replace with the actual gesture name from the model
        # Rotate the second revolute joint forward
        current_value = last_positions[revolute_joint_2]
        new_value = current_value + step_size
        new_value = min(new_value, 3.14)  # Ensure within valid range
        print(f"Revolute Joint 2 rotated forward to {new_value}")
        p.setJointMotorControl2(robot_id, joint_ids[revolute_joint_2], p.POSITION_CONTROL, targetPosition=new_value)
        last_positions[revolute_joint_2] = new_value  # Update last known position
    elif gesture == "Fist":  # Replace with the actual gesture name from the model
        # Rotate the second revolute joint backward
        current_value = last_positions[revolute_joint_2]
        new_value = current_value - step_size
        new_value = max(new_value, -3.14)  # Ensure within valid range
        print(f"Revolute Joint 2 rotated backward to {new_value}")
        p.setJointMotorControl2(robot_id, joint_ids[revolute_joint_2], p.POSITION_CONTROL, targetPosition=new_value)
        last_positions[revolute_joint_2] = new_value  # Update last known position
    else:
        # No gesture detected, maintain the last known positions
        for i in range(num_joints):
            p.setJointMotorControl2(robot_id, joint_ids[i], p.POSITION_CONTROL, targetPosition=last_positions[i])

    # Step the simulation
    p.stepSimulation()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windowsz
cap.release()
cv2.destroyAllWindows()

