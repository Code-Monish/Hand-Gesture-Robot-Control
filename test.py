import cv2
import numpy as np
import pickle

# === Load Fourier Templates ===
with open("gesture_templates_from_dataset.pkl", "rb") as f:
    gesture_templates = pickle.load(f)

for key in gesture_templates:
    gesture_templates[key] = np.abs(gesture_templates[key])

# === Gray-Based Mask Function ===
def get_hand_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # you may need to tune this value
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    return mask

# === Contour and Descriptor Functions ===
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
    descriptors = descriptors / (np.abs(descriptors[0]) + 1e-8)
    return np.abs(descriptors)

def descriptor_distance(desc1, desc2):
    if desc1 is None or desc2 is None or len(desc1) != len(desc2):
        return np.inf
    return np.linalg.norm(desc1 - desc2)

# === Webcam Loop ===
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not detected.")
        break

    frame = cv2.flip(frame, 1)
    mask = get_hand_mask(frame)
    contour = get_largest_contour(mask)
    predicted_label = "Detecting..."

    if contour is not None and len(contour) > 50:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        current_descriptor = fourier_descriptor(contour)

        if current_descriptor is not None:
            min_dist = float('inf')
            best_label = "none"

            for label, template in gesture_templates.items():
                if len(template) != len(current_descriptor):
                    print(f"[WARNING] Template size mismatch for {label}")
                    continue

                dist = descriptor_distance(current_descriptor, template)
                print(f"[DEBUG] Distance to {label}: {dist:.4f}")
                if dist < min_dist:
                    min_dist = dist
                    best_label = label

            if min_dist < 0.5:
                predicted_label = best_label
            else:
                predicted_label = "Uncertain"
        else:
            print("[DEBUG] Descriptor computation failed")
    else:
        print("[DEBUG] No valid contour found")

    # Display
    cv2.putText(frame, f"Gesture: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

