"""
train.py
- builds dataset embeddings via recognition module
- trains scaler + KNN
- evaluates and saves model.pkl
"""

import os
import cv2
import joblib
import numpy as np
import pandas as pd

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
from recognition import get_faces_and_embeddings_from_bgr

IMAGES_FOLDER = "dataset/Images/"
FACES_FOLDER = "dataset/Faces/"
os.makedirs(FACES_FOLDER, exist_ok=True)

labels_df = pd.read_csv("dataset/Dataset.csv")


def crop_and_save_all():
    """
    Crop and save face crops (use DeepFace.extract_faces internally via recognition).
    This makes debugging & dataset inspection easier.
    """
    for _, row in labels_df.iterrows():
        img_name = row["id"]
        in_path = os.path.join(IMAGES_FOLDER, img_name)
        out_path = os.path.join(FACES_FOLDER, img_name)
        if not os.path.exists(in_path):
            print("Missing:", in_path)
            continue
        if os.path.exists(out_path):
            continue

        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print("Unreadable:", in_path)
            continue

        # Use recognition to detect and return first face crop
        faces = get_faces_and_embeddings_from_bgr(img_bgr, enforce_detection=False)
        if not faces:
            print("No face detected (crop skip):", img_name)
            continue

        # We already got embeddings + box; but we want to store the crop itself:
        # Re-run detection to obtain face crop via DeepFace.extract_faces (works inside recognition)
        # Simpler: reuse the box to crop from original image (safe)
        _, (top, right, bottom, left) = faces[0]
        crop = img_bgr[top:bottom, left:right]
        if crop.size == 0:
            print("Invalid crop area:", img_name)
            continue
        cv2.imwrite(out_path, crop)


def build_embeddings():
    encs = []
    labs = []
    for _, row in labels_df.iterrows():
        img_name = row["id"]
        path = os.path.join(FACES_FOLDER, img_name)
        if not os.path.exists(path):
            # fallback: try detect on original if crop missing
            path = os.path.join(IMAGES_FOLDER, img_name)
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            faces = get_faces_and_embeddings_from_bgr(img_bgr, enforce_detection=False)
        else:
            img_bgr = cv2.imread(path)
            faces = get_faces_and_embeddings_from_bgr(img_bgr, enforce_detection=False)

        if not faces:
            print("No embedding generated for:", img_name)
            continue

        emb, _ = faces[0]  # take first face only
        encs.append(emb)
        labs.append(row["label"])

    return np.array(encs), np.array(labs)


def main():
    print("Cropping faces (optional)...")
    crop_and_save_all()

    print("Building embeddings...")
    X, y = build_embeddings()
    if len(X) == 0:
        raise RuntimeError("No embeddings found. Check dataset and detector.")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="distance")
    knn.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = knn.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="macro"))
    print("Recall:", recall_score(y_test, y_pred, average="macro"))
    print("F1:", f1_score(y_test, y_pred, average="macro"))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Distance threshold
    distances, _ = knn.kneighbors(X_train_scaled, n_neighbors=2)
    nearest = distances[:, 1]
    threshold = nearest.mean() + 2 * nearest.std()
    print("Suggested DISTANCE_THRESHOLD =", threshold)

    # Save model
    joblib.dump({"scaler": scaler, "model": knn}, Config.MODEL_PATH)
    print("Saved model to", Config.MODEL_PATH)


if __name__ == "__main__":
    main()
