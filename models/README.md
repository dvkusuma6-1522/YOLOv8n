# Model Checkpoints & Architecture — IRC 2026 YOLOv8n

This directory documents the neural network architecture, model checkpoints, training hyperparameters, and export formats.

---

## 🧠 Model Architecture Specifications

- **Architecture**: Ultralytics YOLOv8 Nano (`yolov8n`)
- **Total Layers**: 73 fused layers
- **Total Parameters**: 3,006,038 (~3.0M parameters)
- **Computational Complexity**: 8.1 GFLOPs
- **Input Resolution**: $640 \times 640 \times 3$
- **Target Classes**:
  - `0: cone`
  - `1: circle` (unifying geometric circles and white chalk circles)

---

## ⚡ Inference Speed Benchmark (NVIDIA Tesla T4 GPU)

| Stage | Latency per Image | Throughput |
| :--- | :---: | :---: |
| **Preprocessing** | 0.2 ms | — |
| **Model Inference (FP16)** | 2.4 ms | ~416 FPS |
| **Postprocessing (NMS)** | 3.5 ms | — |
| **Total End-to-End** | **6.1 ms** | **~164 FPS** |

*Edge Performance*: Tested on Jetson Orin Nano with TensorRT FP16, achieving sustained **>45 FPS** real-time rover perception.

---

## 🗄️ Checkpoint Registry

Checkpoints are saved during Google Colab training and synchronized to Google Drive:

| Checkpoint Name | Format | Size | Description | Google Drive Location |
| :--- | :---: | :---: | :--- | :--- |
| **`IRC2026_YOLOv8n_best.pt`** | PyTorch | ~6.2 MB | Weights with highest validation mAP@50-95 | `IRC2026_Rover_Vision/models/` |
| **`IRC2026_YOLOv8n_last.pt`** | PyTorch | ~6.2 MB | Checkpoint from final epoch (resumption capable) | `IRC2026_Rover_Vision/models/` |
| **`IRC2026_YOLOv8n.onnx`** | ONNX | ~12.1 MB | Slimmed ONNX graph (opset 12) for TensorRT / ROS2 | `IRC2026_Rover_Vision/models/` |

---

## 🚀 Loading Checkpoints in Python

### Ultralytics PyTorch
```python
from ultralytics import YOLO

# Load best checkpoint
model = YOLO("IRC2026_YOLOv8n_best.pt")

# Predict on an image
results = model.predict(source="test_rover.jpg", conf=0.25, iou=0.45)
for box in results[0].boxes:
    print(f"Class: {model.names[int(box.cls[0])]}, Conf: {float(box.conf[0]):.2f}")
```

### ONNX Runtime Inference
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("IRC2026_YOLOv8n.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})
```
