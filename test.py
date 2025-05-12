import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


# Parameters
image_size = (64, 64)  # Resize images
data_dir = r"data\test" # Folder with subfolders for each class

X = []
y = []

# Step 1: Load and preprocess images
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, image_size)
        img = img / 255.0  # Normalize
        X.append(img.flatten())  # Flatten to 1D
        y.append(label)

X = np.array(X)
y = np.array(y)

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train KNN
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
joblib.dump(knn, "knn_model.pkl")

# Step 4: Evaluate
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
