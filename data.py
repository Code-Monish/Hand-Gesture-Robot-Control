import os
import cv2

# Use raw string (prefix with r) to avoid backslash errors
input_folder = r"C:\Users\Hemanth\Desktop\Hand gesture\data\data\train\thumbs"
output_folder = r"C:\Users\Hemanth\Desktop\Hand gesture\data\data\train_padded\thumbs"
pad_width = 10  # Change as needed

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through image files
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Pad with zeros
        padded = cv2.copyMakeBorder(image, pad_width, pad_width, pad_width, pad_width,
                                    borderType=cv2.BORDER_CONSTANT, value=0)

        # Save padded image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, padded)

print("✅ Done: Padded images saved to:", output_folder)
