"""
live_detection.py
Real-time attendance pipeline.
"""

import cv2
import base64
import joblib
import threading
import os

from config import Config
from dashboard import run_dashboard
from attendance import mark_seen
from recognition import get_faces_and_embeddings_bgr


def start_dashboard():
    threading.Thread(target=run_dashboard, daemon=True).start()


def load_pipeline():
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError("Missing model.pkl")
    return joblib.load(Config.MODEL_PATH)


def predict_name(pipe, emb):
    X = pipe["scaler"].transform([emb])
    dist, _ = pipe["model"].kneighbors(X, n_neighbors=1)
    d = float(dist[0][0])

    if d > Config.DISTANCE_THRESHOLD:
        return "Unknown", d

    pred = pipe["model"].predict(X)[0]
    return pred, d


def main():
    start_dashboard()
    pipe = load_pipeline()

    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera error")
        return

    frame_count = 0
    last_results = []

    print("Press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()
        h, w = frame.shape[:2]

        scale = Config.FRAME_WIDTH / w if w > Config.FRAME_WIDTH else 1.0
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        if frame_count % Config.PROCESS_EVERY_N == 0:
            faces = get_faces_and_embeddings_bgr(small, enforce_detection=False)
            results = []

            for emb, face_rgb, (t, r, b, l) in faces:

                t = int(t / scale)
                l = int(l / scale)
                r = int(r / scale)
                b = int(b / scale)

                name, dist = predict_name(pipe, emb)
                results.append((name, t, r, b, l, dist))

                if name != "Unknown":
                    crop = original[t:b, l:r]
                    if crop.size > 0:
                        _, buf = cv2.imencode(".jpg", crop)
                        mark_seen(name, base64.b64encode(buf).decode())

            last_results = results

        for name, t, r, b, l, dist in last_results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({dist:.2f})" if name != "Unknown" else "Unknown"
            cv2.rectangle(original, (l, t), (r, b), color, 2)
            cv2.putText(
                original, label, (l, t - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        cv2.imshow("Attendance", original)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
