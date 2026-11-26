# 🚀 IRC 2026 — Autonomous Rover Object Detection (YOLOv8n)

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/dvkusuma6-1522/YOLOv8n)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-00FFFF?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-NVIDIA%20T4-F9AB00?logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

Unified, real-time two-class object detection pipeline engineered for the **Indian Rover Challenge (IRC) 2026** autonomous navigation subsystem.

---

## 🎯 Competition Mission & Target Classes

Autonomous rovers operating in desert and gravel terrain must reliably identify boundary markers and target waypoints under extreme lighting variations, dust, and partial occlusions.

This model provides a unified detector for:
- **`Class 0: cone`** — Traffic cones, marker cones, and boundary cones across varied angles, orientations, and distances.
- **`Class 1: circle`** — Geometric circular shapes, colored circular targets, and **faint white chalk circles drawn on natural outdoor gravel/dirt terrain**, including chalk circles co-occurring alongside cones.

---

## 📊 Benchmark Evaluation Metrics (NVIDIA Tesla T4 GPU)

The model was trained for 25 epochs using Ultralytics YOLOv8 Nano (`yolov8n.pt`) with Automatic Mixed Precision (AMP / FP16):

| Class | Precision (P) | Recall (R) | mAP @ 0.50 | mAP @ 0.50:0.95 | Inference Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall (`all`)** | **90.8%** | **84.1%** | **91.0%** | **67.3%** | **2.4 ms / image** |
| **`cone` (Class 0)** | 88.5% | 87.5% | 90.8% | 60.2% | — |
| **`circle` (Class 1)** | 93.1% | 80.7% | 91.3% | 74.4% | — |

- **GPU Throughput**: ~**164 FPS** end-to-end (preprocess + inference + NMS) on NVIDIA Tesla T4.
- **Edge Deployment**: Sustained **>45 FPS** on NVIDIA Jetson Orin Nano with TensorRT FP16.

---

## 📁 Repository Structure

```
YOLOv8n/
│
├── README.md                                # Comprehensive project & deployment guide
├── IRC2026_YOLOv8n_Object_Detection.ipynb   # Complete, audited Google Colab notebook
├── requirements.txt                         # Production Python dependencies
├── .gitignore                               # Git configuration excluding large binaries
│
├── datasets/
│   ├── README.md                            # Dataset architecture & balancing summary
│   ├── dataset_sources.md                   # Roboflow Universe provenance & class mappings
│   └── raw/
│       ├── cone_detector/                   # Configs for 21-jyqut/cone-detector-uexft (v1)
│       │   ├── data.yaml
│       │   └── README.md
│       ├── circle_dataset/                  # Configs for deneme-3usbo/circle-9po6m (v1)
│       │   ├── data.yaml
│       │   └── README.md
│       └── white_chalk_circle_and_cone/     # Configs for kusuma-dv/white-chalk-circle-and-cone (v3)
│           ├── data.yaml
│           └── README.md
│
├── models/
│   └── README.md                            # Checkpoint specifications, latency, and loading guide
│
├── results/
│   ├── README.md                            # Evaluation reports & metric graphs
│   ├── results.png                          # Training loss and mAP convergence curves
│   ├── confusion_matrix.png                 # Final validation confusion matrix
│   ├── BoxF1_curve.png                      # F1-Confidence curve (optimal threshold ~0.25-0.45)
│   └── sample_predictions/                  # Real-world inference sample detections
│       ├── sample_cone_detection.png
│       └── sample_chalk_circle_detection.png
│
└── rover_training_data/                     # Complete extracted Google Drive project folder (857.8 MB)
    └── IRC2026_Rover_Vision/
        ├── datasets/
        │   ├── merged/
        │   └── raw/                         # Raw dataset archives (cone, circle, white-chalk)
        ├── models/                          # Trained weights (.pt, .onnx, data.yaml)
        └── results/                         # Training run outputs, epoch checkpoints, batch mosaics
```

---

## 🔬 Dataset Engineering & Preprocessing Pipeline

The model fuses three specialized Roboflow datasets totaling **9,804 images** and **21,233 annotations**:

1. **Cone Detector** (`21-jyqut / cone-detector-uexft` v1): 6,620 images of orange traffic cones.
2. **Circle Dataset** (`deneme-3usbo / circle-9po6m` v1): 2,171 images of multi-colored geometric circles.
3. **White Chalk Circle and Cone** (`kusuma-dv / white-chalk-circle-and-cone` v3): 1,013 images featuring real-world chalk ground circles and cones on rover competition terrain.

### Key Preprocessing Algorithms:
- **Roboflow Polygon-to-BBox Conversion**: Converts multi-vertex segmentation polygons ($>5$ tokens) into tight rectangular bounding box envelopes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \rightarrow [x_c, y_c, w, h]$.
- **Safe Coordinate Clipping**: Strict boundary clipping to $[0.0, 1.0]$, preventing dropped objects caused by data augmentation bleed.
- **Cross-Split Data Leakage Prevention**: SHA256 image hashes of all validation and test images are indexed; any training image matching an evaluation hash is automatically eliminated.
- **Train-Only Class Balancing**: Corrects the raw **4.71 : 1** cone-to-circle training imbalance by oversampling minority circle images ($2\times$) and subsampling redundant pure-cone images to reach **~1.24 : 1**. **Validation and test splits remain 100% untouched.**

---

## 📈 Visual Training & Validation Results

| Training Loss & mAP Curves | Confusion Matrix |
| :---: | :---: |
| ![Training Curves](results/results.png) | ![Confusion Matrix](results/confusion_matrix.png) |

| F1-Confidence Curve | Sample Rover Prediction |
| :---: | :---: |
| ![F1 Curve](results/BoxF1_curve.png) | ![Sample Prediction](results/sample_predictions/sample_chalk_circle_detection.png) |

---

## 🛠️ Google Colab Execution Guide

### Step 1: Open Notebook in Google Colab
Open [`IRC2026_YOLOv8n_Object_Detection.ipynb`](IRC2026_YOLOv8n_Object_Detection.ipynb) in Google Colab.

### Step 2: Set Up Roboflow API Key in Colab Secrets
1. Click the **Key icon** (Secrets) on the left sidebar of Google Colab.
2. Add a new secret:
   - **Name**: `ROBOFLOW_API_KEY`
   - **Value**: *Your private Roboflow API key*
3. Enable **Notebook access**.

### Step 3: Enable NVIDIA T4 GPU Acceleration
1. Navigate to **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.

### Step 4: Run All Cells
Run all cells sequentially:
- Raw datasets are downloaded and cached to Google Drive.
- Model trains with early stopping (`patience=15`), FP16 AMP, and periodic Drive checkpointing every 5 epochs.
- Evaluates on the independent test set and exports to ONNX.
- **Section 11** allows you to upload custom rover camera images to test inference interactively.

---

## 🤖 Rover Edge Deployment (ROS2 & TensorRT)

The exported ONNX model (`IRC2026_YOLOv8n.onnx`) is optimized for embedded edge deployment:

### Jetson TensorRT Engine Compilation
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=IRC2026_YOLOv8n.onnx \
  --saveEngine=IRC2026_YOLOv8n.engine \
  --fp16 \
  --workspace=2048
```

### ROS2 Python Node Integration
```python
from ultralytics import YOLO
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Load TensorRT engine or ONNX model directly in ROS2
model = YOLO("IRC2026_YOLOv8n.engine", task="detect")
bridge = CvBridge()

def camera_callback(msg):
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    results = model.predict(source=frame, conf=0.35, imgsz=640, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected {model.names[cls_id]} with confidence {conf:.2f}")
```

---

## 🔒 Security & Credential Hygiene
- **Zero API Key Leakage**: No hardcoded API keys are present in any notebook cells or repository files.
- **Safe Secrets Usage**: Roboflow API credentials are exclusively retrieved at runtime via `google.colab.userdata.get('ROBOFLOW_API_KEY')` with interactive masked fallback.

---

## 📦 Complete Rover Training Dataset and Model Artifacts

This repository includes the complete extracted Google Drive training directory under [`rover_training_data/IRC2026_Rover_Vision/`](rover_training_data/IRC2026_Rover_Vision/) (857.75 MB across 39 files), preserving all original datasets, checkpoints, training batch mosaics, and evaluation logs.

### 1. Folder Structure & Contents

```
rover_training_data/IRC2026_Rover_Vision/
├── datasets/
│   ├── merged/                              # Staging directory for balanced unified dataset
│   └── raw/                                 # Complete raw Roboflow source archives
│       ├── cone-detector-uexft-1.zip        # 453.10 MB (6,620 images of traffic cones)
│       ├── circle-9po6m-1.zip               #  99.60 MB (2,171 images of geometric circles)
│       └── white-chalk-circle-and-cone-3.zip #  59.97 MB (1,013 images of chalk circles & cones)
│
├── models/                                  # Exported weights & deployment models
│   ├── IRC2026_YOLOv8n_best.pt              # 6.20 MB (Best validation mAP PyTorch weights)
│   ├── IRC2026_YOLOv8n_last.pt              # 6.20 MB (Final epoch checkpoint for resumption)
│   ├── IRC2026_YOLOv8n_best_20261125_233233.pt # 6.20 MB (Timestamped best checkpoint)
│   ├── IRC2026_YOLOv8n.onnx                 # 12.18 MB (Slimmed ONNX model for TensorRT/ROS 2)
│   ├── IRC2026_YOLOv8n_best.onnx            # 12.18 MB (Verified ONNX best model)
│   └── IRC2026_data.yaml                    # Unified dataset configuration
│
└── results/
    └── run_20261125_233233/
        └── training_run/                    # Full training run outputs (25 epochs on Tesla T4)
            ├── args.yaml                    # Exact Ultralytics training hyperparameters
            ├── results.csv                  # Epoch-by-epoch loss and mAP progression
            ├── results.png                  # Loss convergence & mAP curves
            ├── confusion_matrix.png         # Unnormalized confusion matrix
            ├── confusion_matrix_normalized.png # Normalized confusion matrix
            ├── BoxF1_curve.png              # F1 vs Confidence curve
            ├── BoxPR_curve.png              # Precision-Recall curve
            ├── BoxP_curve.png               # Precision vs Confidence curve
            ├── BoxR_curve.png               # Recall vs Confidence curve
            ├── labels.jpg                   # Training label distribution & bounding box spatial map
            ├── train_batch*.jpg             # Mosaic augmented training batches (6 mosaic images)
            ├── val_batch*_labels.jpg        # Ground-truth validation targets (3 batches)
            ├── val_batch*_pred.jpg          # Model prediction overlays on validation batches (3 batches)
            └── weights/                     # Checkpoints saved during training
                ├── best.pt                  # 6.20 MB
                ├── last.pt                  # 6.20 MB
                ├── epoch0.pt                # 30.31 MB (Includes optimizer state)
                ├── epoch5.pt                # 30.31 MB
                ├── epoch10.pt               # 30.32 MB
                ├── epoch15.pt               # 30.32 MB
                ├── epoch20.pt               # 30.32 MB
                └── epoch25.pt               # 30.32 MB
```

### 2. How to Access Large Files (Git LFS)

Large binary files (`.pt`, `.onnx`, and `.zip` archives) are tracked seamlessly via **Git Large File Storage (Git LFS)**:

```bash
# Clone with LFS objects
git lfs install
git clone https://github.com/dvkusuma6-1522/YOLOv8n.git

# Or pull LFS binaries in an existing clone
git lfs pull
```

### 3. How to Use the Trained Models

#### Direct Python / Ultralytics Inference
```python
from ultralytics import YOLO

# Load best PyTorch model
model = YOLO("rover_training_data/IRC2026_Rover_Vision/models/IRC2026_YOLOv8n_best.pt")

# Predict on rover camera frame
results = model.predict(source="test_frame.jpg", conf=0.25, iou=0.45, imgsz=640)
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()
    print(f"Detected {model.names[cls_id]} ({conf*100:.1f}%) at {xyxy}")
```

#### ONNX Runtime Inference (Edge / ROS 2)
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "rover_training_data/IRC2026_Rover_Vision/models/IRC2026_YOLOv8n.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
input_name = session.get_inputs()[0].name
dummy_img = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = session.run(None, {input_name: dummy_img})
```

### 4. Reproducing Training from Scratch
To retrain using the raw datasets in `rover_training_data/`:
1. Unzip `datasets/raw/*.zip` into local SSD scratch storage.
2. Open [`IRC2026_YOLOv8n_Object_Detection.ipynb`](IRC2026_YOLOv8n_Object_Detection.ipynb) in Google Colab.
3. Execute all cells to perform polygon-to-bbox conversion, class balancing, and T4 GPU fine-tuning.

