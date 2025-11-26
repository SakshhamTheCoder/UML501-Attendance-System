class Config:
    # CAMERA CONFIG
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    PROCESS_EVERY_N = 3  # Run detection every N frames

    # MODEL CONFIG
    MODEL_PATH = "model.pkl"
    DETECTOR_BACKEND = "retinaface"  # deepface detector
    EMBEDDING_MODEL = "ArcFace"  # deepface embedding model

    # CLASSIFIER CONFIG
    DISTANCE_THRESHOLD = 25.16  # tuned threshold
    ENCODING_NORMALIZE = False  # not needed for DeepFace embeddings

    # DASHBOARD CONFIG
    SSE_UPDATE_INTERVAL = 1
    DEBOUNCE_SECONDS = 3
