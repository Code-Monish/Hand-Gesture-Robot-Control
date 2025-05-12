import cv2
import joblib
import numpy as np

# Parameters
image_size = (64, 64)  # Same size used during training
model_path = "knn_model.pkl"  # Path to the saved KNN model

# Load the trained KNN model
knn = joblib.load(model_path)

# Function to apply Fourier high-pass filter
def apply_highpass_filter(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Create a high-pass mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # Radius of the low-frequency region to block
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0

    # Apply the mask and inverse DFT
    filtered_dft = dft_shift * mask
    filtered_dft_shift = np.fft.ifftshift(filtered_dft)
    filtered_image = np.fft.ifft2(filtered_dft_shift)
    filtered_image = np.abs(filtered_image)

    return filtered_image

# Start capturing live camera feed
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to HSV and segment skin color (basic hand detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Extract the hand region
    hand_region = cv2.bitwise_and(frame, frame, mask=mask)
    gray_hand = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

    # Threshold the hand region to make the hand white and background black
    _, binary_hand = cv2.threshold(gray_hand, 50, 255, cv2.THRESH_BINARY)

    # Apply Fourier high-pass filter
    filtered_hand = apply_highpass_filter(binary_hand)

    # Threshold again to ensure the hand is white and background is black
    _, final_hand = cv2.threshold(filtered_hand, 50, 255, cv2.THRESH_BINARY)

    # Resize and normalize the final hand image
    resized = cv2.resize(final_hand, image_size)
    normalized = resized / 255.0
    flattened = normalized.flatten().reshape(1, -1)

    # Predict the class
    prediction = knn.predict(flattened)
    predicted_label = prediction[0]

    # Display the prediction on the frame
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame and the processed hand region
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Processed Hand", final_hand)  # Display the final binary hand image

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
