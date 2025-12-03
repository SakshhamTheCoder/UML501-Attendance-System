# 📸 Face Attendance System (DeepFace + KNN)

A lightweight, high-performance **real-time face attendance system** using:

-   **DeepFace** for _face detection_, _alignment_, _embeddings_
-   **KNN classifier** for custom identity prediction
-   **OpenCV** for live video processing
-   **Flask** dashboard with live event updates (SSE)
-   Optimized inference loop with asynchronous attendance logging

This project is designed for **speed**, **accuracy**, and **easy customization**, and is suitable for deployments in offices, labs, or small institutions.

---

# 🚀 Features

### 🔹 **1. Real-time Face Recognition**

-   Uses **DeepFace.represent()** → a single call performs detection + alignment + embedding.
-   Highly accurate deep learning embeddings (Facenet / ArcFace / VGGFace supported).
-   Fast KNN classifier for quick prediction.

### 🔹 **2. Live Attendance Dashboard**

-   Browser-based dashboard via Flask.
-   Auto-refresh using **Server-Sent Events (SSE)**.
-   View:

    -   Current in/out status
    -   Live event timeline
    -   Latest face snapshots

### 🔹 **3. Fast & Optimized Pipeline**

-   DeepFace called **only once per N frames**.
-   Async attendance logging prevents bottlenecks.
-   Full-resolution embeddings (not resized → better accuracy).
-   Clean modular architecture.

### 🔹 **4. Customizable**

-   Plug in different embedding models.
-   Replace KNN with SVM / RandomForest / Neural Network.
-   Adjustable thresholds, debounce time, camera index, etc.

---

# 🏗 Project Structure

```
.
├── live_detection.py     # Real-time face recognition + dashboard updates
├── train.py              # Train KNN with embeddings from DeepFace
├── recognition.py        # Unified face detection + embedding pipeline
├── attendance.py         # In/out logic + event management
├── dashboard.py          # Flask dashboard with SSE updates
├── config.py             # Centralized configuration
├── dataset/
│   ├── Images/           # Original user photos
│   ├── Faces/            # Auto-cropped face images
│   └── Dataset.csv       # {id, label} mapping
└── model.pkl             # Saved scaler + KNN classifier
```

---

# ⚙️ Installation

### **1. Install Python dependencies**

```bash
pip install -r requirements.txt
```

Minimum required packages:

```
deepface
opencv-python
scikit-learn
numpy
pandas
flask
joblib
```

### **2. Install TensorFlow GPU (Optional but recommended)**

This project supports CUDA acceleration:

```bash
pip install "tensorflow[and-cuda]"
```

### **3. Check GPU availability**

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

---

# 🧠 Training the Model

Prepare your dataset:

```
dataset/
├── Images/
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
└── Dataset.csv
    id,label
    person1.jpg,John
    person2.jpg,Aisha
```

Run training:

```bash
python train.py
```

This script:

-   Detects and crops faces
-   Extracts DeepFace embeddings
-   Trains a **KNN classifier**
-   Suggests a distance threshold
-   Saves `model.pkl`

---

# 🎥 Running Live Detection

Start the full system:

```bash
python live_detection.py
```

It will:

-   Open webcam
-   Start Flask dashboard at:

```
http://localhost:5000
```

Press **Q** to exit.

---

# 📡 Dashboard

The dashboard displays:

-   ✔ Current attendance status
-   ✔ Live events (IN / OUT)
-   ✔ Recent face snapshots
-   ✔ Updates in real-time (SSE)

No page refresh required.

---

# ⚙️ Configuration

All important settings live inside `config.py`:

```python
class Config:
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    PROCESS_EVERY_N = 5
    MODEL_PATH = "model.pkl"

    DETECTOR_BACKEND = "retinaface"
    EMBEDDING_MODEL = "Facenet"

    DISTANCE_THRESHOLD = 13.0
    DEBOUNCE_SECONDS = 3
```

Adjust according to your camera or environment.

---
