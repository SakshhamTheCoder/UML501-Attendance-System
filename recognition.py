import numpy as np
import face_recognition
from config import Config


def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def get_encodings(rgb_image):
    boxes = face_recognition.face_locations(rgb_image, model=Config.FACE_DETECT_MODEL)
    if not boxes:
        return [], []

    encs = face_recognition.face_encodings(rgb_image, boxes)
    if Config.ENCODING_NORMALIZE:
        encs = [normalize(e) for e in encs]

    return encs, boxes


def predict(pipeline, encoding):
    """
    Return label or 'Unknown' using KNN distance threshold
    pipeline = {"scaler": scaler, "model": knn}
    """
    X = np.array(encoding).reshape(1, -1)
    X_scaled = pipeline["scaler"].transform(X)

    # nearest neighbor distance
    distances, _ = pipeline["model"].kneighbors(X_scaled, n_neighbors=1)
    closest_dist = distances[0][0]
    print(closest_dist)
    label = pipeline["model"].predict(X_scaled)[0]

    if closest_dist > Config.DISTANCE_THRESHOLD:
        return "Unknown"
    return label
