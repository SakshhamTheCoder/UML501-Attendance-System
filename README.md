# Face Recognition Flask App Structure

## Files:

-   `train_model.py`: Script to train the face recognition model using images in `dataset/Faces/` and labels from `dataset/Dataset.csv`. Saves the trained model to disk.
-   `app.py`: Flask app for video upload and prediction. Loads the trained model and processes uploaded videos.
-   `recognition_utils.py`: Shared utilities for face detection, embedding extraction, and prediction.
-   `requirements.txt`: Python dependencies.

## Workflow:

1. Run `train_model.py` once to train and save the model (e.g., as `model.pkl` or similar).
2. Start the Flask app (`app.py`).
3. Upload a video via the web interface or API.
4. The app extracts frames, detects faces, predicts identities, and returns a summary of recognized people.

---

Proceeding to implement the training script and Flask app.

