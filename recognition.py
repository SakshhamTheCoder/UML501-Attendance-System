"""
recognition.py
- unified detection + embedding functions using DeepFace
- ensures identical preprocessing for training and runtime
"""

import numpy as np
import cv2
from deepface import DeepFace
from config import Config
from typing import List, Tuple, Optional, Dict


def _to_uint8_rgb(face_array: np.ndarray) -> np.ndarray:
    """
    DeepFace.extract_faces returns 'face' as normalized float [0,1] RGB.
    Convert to uint8 RGB (0-255).
    If input already uint8 BGR, convert appropriately before calling functions.
    """
    if face_array.dtype != np.uint8:
        face_rgb = (face_array * 255).astype("uint8")
    else:
        face_rgb = face_array
    return face_rgb


def detect_faces_and_boxes(
    image_rgb: np.ndarray, enforce_detection: bool = False
) -> List[Dict]:
    """
    Run DeepFace.extract_faces on an RGB image and return the raw detection dicts.
    Each detection dict contains:
      - 'face' : numpy array (RGB) of the aligned face cropped by DeepFace
      - 'facial_area' : dict with {x, y, w, h}
      - other metadata
    Returns [] if no faces found.
    Note: image_rgb must be RGB (not BGR).
    """
    try:
        detections = DeepFace.extract_faces(
            img_path=image_rgb,
            detector_backend=Config.DETECTOR_BACKEND,
            enforce_detection=enforce_detection,
            align=True,
        )
        if not isinstance(detections, list):
            return []
        return detections
    except Exception:
        return []


def get_face_crop_from_detection(det: Dict) -> np.ndarray:
    """
    Given one extraction dict from DeepFace.extract_faces, return uint8 RGB face crop.
    """
    face = det.get("face")
    if face is None:
        return None
    return _to_uint8_rgb(face)


def detection_box_from_det(det: Dict) -> Tuple[int, int, int, int]:
    """
    Convert DeepFace detection 'facial_area' to (top, right, bottom, left)
    """
    fa = det.get("facial_area")
    if not isinstance(fa, dict):
        # fallback if different format (x1,y1,x2,y2)
        try:
            x1, y1, x2, y2 = fa
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
        except Exception:
            return (0, 0, 0, 0)
    else:
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

    top = int(y)
    left = int(x)
    right = int(x + w)
    bottom = int(y + h)
    return (top, right, bottom, left)


def generate_embedding_from_face_rgb(face_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Generate a single embedding for a face (RGB uint8 image) using DeepFace.represent.
    detector_backend='skip' because we already have the crop.
    Returns numpy array embedding or None on failure.
    """
    try:
        # DeepFace.represent accepts path or ndarray; when passing ndarray, it expects RGB values 0-255
        rep = DeepFace.represent(
            img_path=face_rgb,
            model_name=Config.EMBEDDING_MODEL,
            detector_backend="skip",
        )
        if not rep or not isinstance(rep, list):
            return None
        emb = np.array(rep[0]["embedding"], dtype=np.float32)
        return emb
    except Exception:
        return None


def get_faces_and_embeddings_from_rgb(
    image_rgb: np.ndarray, enforce_detection: bool = False
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    High-level function used by training and runtime:
      - input: RGB image (numpy array)
      - returns: list of (embedding, (top,right,bottom,left))
    This uses DeepFace.extract_faces -> generate_embedding_from_face_rgb
    """
    outputs = []
    detections = detect_faces_and_boxes(image_rgb, enforce_detection=enforce_detection)
    for det in detections:
        face_rgb = get_face_crop_from_detection(det)
        if face_rgb is None:
            continue

        emb = generate_embedding_from_face_rgb(face_rgb)
        if emb is None:
            continue

        box = detection_box_from_det(det)
        outputs.append((emb, box))
    return outputs


# convenience wrapper for BGR OpenCV frames
def get_faces_and_embeddings_from_bgr(
    image_bgr: np.ndarray, enforce_detection: bool = False
):
    """
    Accepts BGR frame (as read by OpenCV), converts to RGB and returns same output format.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return get_faces_and_embeddings_from_rgb(rgb, enforce_detection=enforce_detection)
