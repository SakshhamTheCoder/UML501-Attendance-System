import os
import pandas as pd
import face_recognition
import joblib
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from config import Config


def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


IMAGES_FOLDER = "dataset/Images/"
FACES_FOLDER = "dataset/Faces/"
os.makedirs(FACES_FOLDER, exist_ok=True)

labels_df = pd.read_csv("dataset/Dataset.csv")

# Crop faces
for _, row in labels_df.iterrows():
    img_name = row["id"]
    input_path = os.path.join(IMAGES_FOLDER, img_name)
    output_path = os.path.join(FACES_FOLDER, img_name)

    if not os.path.exists(input_path) or os.path.exists(output_path):
        continue

    image = face_recognition.load_image_file(input_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        continue

    top, right, bottom, left = face_locations[0]
    face_crop = image[top:bottom, left:right]

    face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

    cropped_path = os.path.join(FACES_FOLDER, img_name)
    cv2.imwrite(cropped_path, face_bgr)

# Generate encodings
encodings, labels = [], []
for _, row in labels_df.iterrows():
    img_path = os.path.join(FACES_FOLDER, row["id"])
    if not os.path.exists(img_path):
        continue

    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        continue

    encoding = face_recognition.face_encodings(image, face_locations)[0]
    encodings.append(normalize(encoding))
    labels.append(row["label"])

if not encodings:
    raise RuntimeError("No encodings found.")

X = np.array(encodings)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn.fit(X_train_scaled, y_train)

# Evaluate
y_pred = knn.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall:", recall_score(y_test, y_pred, average="macro"))
print("F1:", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Distance threshold
distances, _ = knn.kneighbors(X_train_scaled, n_neighbors=2)
nearest_distances = distances[:, 1]
threshold = nearest_distances.mean() + 2 * nearest_distances.std()
print("DISTANCE_THRESHOLD =", round(threshold, 3))

# Save model
joblib.dump({"scaler": scaler, "model": knn}, Config.MODEL_PATH)
