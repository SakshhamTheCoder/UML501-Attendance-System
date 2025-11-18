import cv2
import threading
import os
import joblib
import base64
from config import Config
from recognition import get_encodings, predict
from attendance import mark_seen
from dashboard import run_dashboard


def start_dashboard():
    t = threading.Thread(target=run_dashboard, daemon=True)
    t.start()
    print("Dashboard running @ http://127.0.0.1:5000")


def load_model():
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {Config.MODEL_PATH}")
    return joblib.load(Config.MODEL_PATH)


def main():
    start_dashboard()
    pipeline = load_model()

    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera error")
        return

    frame_count = 0
    last_detections = []

    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()
        h, w = frame.shape[:2]
        scale = Config.FRAME_WIDTH / w if w > Config.FRAME_WIDTH else 1.0
        frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        if frame_count % Config.PROCESS_EVERY_N == 0:
            encs, boxes = get_encodings(rgb_small)
            detections = []

            for enc, (top, right, bottom, left) in zip(encs, boxes):
                name = predict(pipeline, enc)
                t, r, b, l = (
                    int(top / scale),
                    int(right / scale),
                    int(bottom / scale),
                    int(left / scale),
                )
                detections.append((name, (t, r, b, l)))

                if name != "Unknown":
                    # crop face from the original (BGR) frame and encode to JPEG base64
                    try:
                        face_bgr = original[t:b, l:r]
                        # ensure non-empty
                        if face_bgr.size != 0:
                            _, buf = cv2.imencode(".jpg", face_bgr)
                            img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        else:
                            img_b64 = None
                    except Exception:
                        img_b64 = None

                    mark_seen(name, image_b64=img_b64)

            last_detections = detections.copy()
        else:
            detections = last_detections

        for name, (t, r, b, l) in detections:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(original, (l, t), (r, b), color, 2)
            cv2.putText(
                original, name, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
        cv2.namedWindow("Face Attendance", cv2.WINDOW_NORMAL)
        cv2.imshow("Face Attendance", original)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
