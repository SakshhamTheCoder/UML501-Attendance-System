"""
live_detection.py - real-time webcam attendance using recognition module
"""

import cv2
import threading
import joblib
import base64
import os

from config import Config
from recognition import get_faces_and_embeddings_from_bgr
from attendance import mark_seen
from dashboard import run_dashboard


def start_dashboard():
    t = threading.Thread(target=run_dashboard, daemon=True)
    t.start()


def load_pipeline():
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError("model not found: " + Config.MODEL_PATH)
    return joblib.load(Config.MODEL_PATH)


def predict_name(pipeline, embedding):
    X = embedding.reshape(1, -1)
    Xs = pipeline["scaler"].transform(X)
    distances, _ = pipeline["model"].kneighbors(Xs, n_neighbors=1)
    closest = float(distances[0][0])
    print("Closest distance:", closest)
    if closest > Config.DISTANCE_THRESHOLD:
        return "Unknown", closest
    return pipeline["model"].predict(Xs)[0], closest


def main():
    start_dashboard()
    pipeline = load_pipeline()

    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera error")
        return

    frame_count = 0
    last_detections = []

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()
        h, w = frame.shape[:2]
        scale = Config.FRAME_WIDTH / w if w > Config.FRAME_WIDTH else 1.0
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        # small is BGR
        faces = last_detections

        if frame_count % Config.PROCESS_EVERY_N == 0:
            results = get_faces_and_embeddings_from_bgr(small, enforce_detection=False)
            detections = []
            for emb, (top, right, bottom, left) in results:
                name, dist = predict_name(pipeline, emb)
                # rescale coords
                t = int(top / scale)
                r = int(right / scale)
                b = int(bottom / scale)
                l = int(left / scale)
                detections.append((name, (t, r, b, l), dist))

                if name != "Unknown":
                    crop = original[t:b, l:r]
                    if crop.size > 0:
                        _, buf = cv2.imencode(".jpg", crop)
                        img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        mark_seen(name, image_b64=img_b64)

            last_detections = detections

        # draw boxes
        for det in last_detections:
            name, (t, r, b, l), dist = det
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(original, (l, t), (r, b), color, 2)
            label = f"{name} ({dist:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(
                original, label, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        cv2.imshow("Face Attendance", original)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
