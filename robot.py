import pybullet as p
import pybullet_data
import time

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

for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    sliders.append(p.addUserDebugParameter(joint_name, -3.14, 3.14, 0))
    joint_ids.append(i)

# Loop to keep simulation running
while True:
    for i, slider in enumerate(sliders):
        target_pos = p.readUserDebugParameter(slider)
        p.setJointMotorControl2(robot_id, joint_ids[i], p.POSITION_CONTROL, targetPosition=target_pos)
    
    p.stepSimulation()

