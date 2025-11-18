class Config:
    CAMERA_INDEX = 0
    MODEL_PATH = "model.pkl"
    FACE_DETECT_MODEL = "hog"  # "hog" (fast) or "cnn" (accurate, GPU only)
    FRAME_WIDTH = 640
    PROCESS_EVERY_N = 3

    # DISTANCE_THRESHOLD = 9.642  # Distance required to accept identity
    DISTANCE_THRESHOLD = 15.421
    ENCODING_NORMALIZE = True  # L2 normalize encodings for stability

    SSE_UPDATE_INTERVAL = 1  # dashboard update frequency
    DEBOUNCE_SECONDS = 3  # seconds to debounce IN/OUT toggles
