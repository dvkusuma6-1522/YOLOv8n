# Dataset Sources & Provenance Registry

This registry documents all source datasets integrated into the unified YOLOv8n object detection model for the **Indian Rover Challenge (IRC) 2026**.

---

## 🎯 Target Unified Benchmark Classes
- **`0: cone`** — Orange/red traffic cones and navigation markers across natural outdoor environments.
- **`1: circle`** — Geometric circles, colored circular shapes, and competition white chalk circles drawn on terrain.

---

## 📦 Source Datasets Registry

| Dataset ID | Workspace | Project | Version | License | Source Classes | Target Mapping |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Dataset 1** | `21-jyqut` | `cone-detector-uexft` | `v1` | CC BY 4.0 | `cone detection - v1 2024-08-13 3-16pm` (0) | `cone` $\rightarrow$ `0` |
| **Dataset 2** | `deneme-3usbo` | `circle-9po6m` | `v1` | CC BY 4.0 | `Circle` (0), `blue circle` (1), `green circle` (2), `redcircle` (3) | All circles $\rightarrow$ `1` |
| **Dataset 3** | `kusuma-dv` | `white-chalk-circle-and-cone` | `v3` | CC BY 4.0 | `cone` (0), `white_chalk_circle` (1) | `cone` $\rightarrow$ `0`, `circle` $\rightarrow$ `1` |

---

## 📥 Automated Roboflow Ingestion Code

In the primary Colab notebook (`IRC2026_YOLOv8n_Object_Detection.ipynb`), datasets are downloaded securely using Google Colab Secrets without hardcoding credentials:

```python
from roboflow import Roboflow
from google.colab import userdata

rf_api_key = userdata.get('ROBOFLOW_API_KEY')
rf = Roboflow(api_key=rf_api_key)

# 1. Cone Detector
p1 = rf.workspace("21-jyqut").project("cone-detector-uexft").version(1).download("yolov8")

# 2. Circle Dataset
p2 = rf.workspace("deneme-3usbo").project("circle-9po6m").version(1).download("yolov8")

# 3. White Chalk Circle & Cone
p3 = rf.workspace("kusuma-dv").project("white-chalk-circle-and-cone").version(3).download("yolov8")
```

---

## 🔄 Data Harmonization & Leak Prevention Summary

1. **Polygon to Bounding Box**: Roboflow segmentation polygons ($>5$ tokens) are converted to tight bounding box envelopes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \rightarrow [x_c, y_c, w, h]$.
2. **Boundary Clipping**: Augmented out-of-bounds coordinates are clipped to $[0.0, 1.0]$.
3. **Cross-Split Leak Prevention**: SHA256 image hashes of all validation and test images are indexed; any matching training images are excluded to prevent test set contamination.
4. **Train-Only Class Balancing**: Minority circle images are duplicated ($2\times$) and redundant pure-cone images are subsampled in the `train` split only. **Validation and test splits remain untouched.**
