import cv2
import numpy as np
from deepface import DeepFace
from config import Config


def get_faces_and_embeddings_bgr(bgr_image, enforce_detection=False, max_faces=1):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    try:
        reps = DeepFace.represent(
            img_path=rgb,
            model_name=Config.EMBEDDING_MODEL,
            detector_backend=Config.DETECTOR_BACKEND,
            enforce_detection=enforce_detection,
            align=True,
            normalization="base",
            max_faces=max_faces,
        )
    except Exception:
        return []

    if not isinstance(reps, list):
        return []

    faces = []
    for rep in reps:
        emb = np.array(rep["embedding"], dtype=np.float32)

        fa = rep["facial_area"]
        x = int(fa["x"])
        y = int(fa["y"])
        w = int(fa["w"])
        h = int(fa["h"])

        top = y
        left = x
        bottom = y + h
        right = x + w

        face_rgb = rep.get("face")
        faces.append((emb, face_rgb, (top, right, bottom, left)))

    return faces
