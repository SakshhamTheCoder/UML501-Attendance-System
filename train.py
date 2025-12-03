"""
train.py
- Builds dataset embeddings using ONE DeepFace.represent() call
- Saves crops
- Trains KNN
- Saves confusion matrix + performance plots
"""

import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from recognition import get_faces_and_embeddings_bgr
from config import Config


IMAGES = "dataset/Images/"
FACES = "dataset/Faces/"
os.makedirs(FACES, exist_ok=True)

df = pd.read_csv("dataset/Dataset.csv")


def crop_and_save_all():
    print("Cropping...")

    for i, row in df.iterrows():
        img_name = row["id"]
        in_path = os.path.join(IMAGES, img_name)
        out_path = os.path.join(FACES, img_name)

        if os.path.exists(out_path):
            continue

        img = cv2.imread(in_path)
        if img is None:
            print("Unreadable:", img_name)
            continue

        faces = get_faces_and_embeddings_bgr(img, enforce_detection=False)
        if not faces:
            print("No face:", img_name)
            continue

        _, face_rgb, (t, r, b, l) = faces[0]

        crop = img[t:b, l:r]
        if crop.size > 0:
            cv2.imwrite(out_path, crop)


def build_embeddings():
    X, y = [], []
    total = len(df)

    for i, row in df.iterrows():
        img_name = row["id"]

        face_path = os.path.join(FACES, img_name)
        img_path = (
            face_path if os.path.exists(face_path) else os.path.join(IMAGES, img_name)
        )

        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = get_faces_and_embeddings_bgr(img, enforce_detection=False)
        if not faces:
            print("Skip:", img_name)
            continue

        emb, _, _ = faces[0]
        X.append(emb)
        y.append(row["label"])

        print(f"[{i+1}/{total}] {img_name}")

    return np.array(X), np.array(y)


# def save_confusion_matrix(y_true, y_pred, labels):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
#     )
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.tight_layout()
#     plt.savefig("confusion_matrix.png")
#     plt.close()
#     print("Saved confusion_matrix.png")


# def save_performance_plot(acc, prec, rec, f1):
#     metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
#     values = [acc, prec, rec, f1]

#     plt.figure(figsize=(8, 6))
#     sns.barplot(x=metrics, y=values, palette="viridis")
#     plt.ylim(0, 1)
#     plt.title("Model Performance Metrics")
#     plt.tight_layout()
#     plt.savefig("model_performance.png")
#     plt.close()
#     print("Saved model_performance.png")


def main():
    crop_and_save_all()

    print("Building embeddings...")
    X, y = build_embeddings()
    if len(X) == 0:
        raise RuntimeError("No embeddings found.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="distance")
    knn.fit(X_train_s, y_train)

    preds = knn.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    # Save metrics as images
    # save_confusion_matrix(y_test, preds, labels=np.unique(y))
    # save_performance_plot(acc, prec, rec, f1)

    distances, _ = knn.kneighbors(X_train_s, n_neighbors=2)
    threshold = distances[:, 1].mean() + distances[:, 1].std()
    print("Suggested DISTANCE_THRESHOLD =", threshold)

    joblib.dump({"scaler": scaler, "model": knn}, Config.MODEL_PATH)
    print("Saved:", Config.MODEL_PATH)


if __name__ == "__main__":
    main()
