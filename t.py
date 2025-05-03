import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# === PARAMETERS ===
img_size = 64
data_dir = 'C:\Users\Hemanth\Desktop\Hand gesture\data\data\train\fist'  # Folder with subfolders per gesture
gesture_labels = sorted(os.listdir(data_dir))  # Label names from folder structure

# === STEP 1: Load and Preprocess Dataset ===
X, y = [], []

for idx, gesture in enumerate(gesture_labels):
    gesture_path = os.path.join(data_dir, gesture)
    if not os.path.isdir(gesture_path):
        continue
    for img_file in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(idx)

X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 2: Build CNN Model ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === STEP 3: Train the Model ===
print("Training model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# === STEP 4: Evaluate Model ===
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# === STEP 5: Function to Simulate Robot Arm Control ===
def control_robot(class_id):
    if class_id == 0:
        print("Move robot to Home Position")
    elif class_id == 1:
        print("Move robot to Pick Position")
    elif class_id == 2:
        print("Move robot to Place Position")
    elif class_id == 3:
        print("Rotate joint 1")
    elif class_id == 4:
        print("Stop movement")
    else:
        print("Gesture not mapped")

# === STEP 6: Real-time Webcam Inference ===
print("Starting webcam for real-time gesture recognition. Press 'q' to quit.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (img_size, img_size))
    roi_input = roi.reshape(1, img_size, img_size, 1) / 255.0

    prediction = model.predict(roi_input, verbose=0)
    class_id = int(np.argmax(prediction))
    gesture_name = gesture_labels[class_id] if class_id < len(gesture_labels) else "Unknown"

    # Display results
    cv2.putText(frame, f'Gesture: {gesture_name}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    # Control robot
    control_robot(class_id)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
