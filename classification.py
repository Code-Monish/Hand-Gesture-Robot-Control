import cv2
import numpy as np
import os
import pickle

def get_hand_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    return mask

def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if contours and len(contours) > 0:
        return max(contours, key=cv2.contourArea)
    return None

def fourier_descriptor(contour, num_descriptors=32):
    contour = contour.squeeze()
    if len(contour.shape) != 2 or contour.shape[0] < num_descriptors:
        return None
    contour_complex = contour[:, 0] + 1j * contour[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    fourier_result = np.fft.fftshift(fourier_result)
    center = len(fourier_result) // 2
    descriptors = fourier_result[center - num_descriptors // 2: center + num_descriptors // 2]
    descriptors = descriptors / (np.abs(descriptors[0]) + 1e-8)  # normalize
    return np.abs(descriptors)

# === Step 2: Process All Images ===

# Define paths for your dataset
dataset_path = r'data\test'
gesture_labels = ['thumbs', 'fist', 'peace', 'okay', 'rad', 'five', 'straight', 'none']
template_data = {}

for label in gesture_labels:
    label_folder = os.path.join(dataset_path, label)
    descriptors_list = []

    # Process all images in the current gesture folder
    for img_name in os.listdir(label_folder):
        img_path = os.path.join(label_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        mask = get_hand_mask(img)
        contour = get_largest_contour(mask)

        if contour is not None and len(contour) > 50:
            desc = fourier_descriptor(contour)
            if desc is not None:
                descriptors_list.append(desc)

    # Store average descriptor for each gesture
    if descriptors_list:
        template_data[label] = np.mean(descriptors_list, axis=0)

# Save the template data to file
with open("gesture_templates_from_dataset.pkl", "wb") as f:
    pickle.dump(template_data, f)

print("[INFO] Fourier descriptors for gestures saved to 'gesture_templates_from_dataset.pkl'")
