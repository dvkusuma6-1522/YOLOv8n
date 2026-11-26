# Datasets Overview — IRC 2026 YOLOv8n

This directory contains metadata, configurations, and documentation for the datasets used to train the real-time two-class object detection model for the **Indian Rover Challenge (IRC) 2026**.

---

## 📁 Directory Structure

```
datasets/
├── README.md               # Dataset overview and setup guidelines
├── dataset_sources.md      # Roboflow provenance, URLs, licenses, and mapping
└── raw/                    # Raw source dataset configs & metadata
    ├── cone_detector/
    │   ├── data.yaml
    │   └── README.md
    ├── circle_dataset/
    │   ├── data.yaml
    │   └── README.md
    └── white_chalk_circle_and_cone/
        ├── data.yaml
        └── README.md
```

---

## 📊 Dataset Statistics & Class Distribution

| Dataset | Raw Train Images | Raw Valid Images | Raw Test Images | Total Images | Raw Boxes | Primary Classes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **cone_detector** | 5,862 | 558 | 200 | 6,620 | 16,646 | Cones |
| **circle_dataset** | 1,878 | 206 | 87 | 2,171 | 3,259 | Circles (multi-color) |
| **white_chalk_circle_and_cone** | 958 | 35 | 20 | 1,013 | 1,328 | White chalk circles & cones |
| **Total Raw** | **8,698** | **799** | **307** | **9,804** | **21,233** | Combined |

---

## ⚖️ Train-Only Class Balancing Algorithm

In the raw datasets, cones outnumbered circles by approximately **4.71 : 1** (~20,064 cones vs ~4,262 circles in training).

To prevent the model from overfitting to cones and failing on faint chalk circles:
1. **Circle Oversampling**: All circle-containing images in `train` are duplicated ($2\times$) with unique filenames (`_baldup1`).
2. **Pure-Cone Subsampling**: Redundant pure-cone images are subsampled until the ratio reaches **~1.24 : 1**.
3. **Background Negative Preservation**: Negative images (empty labels) are retained at ~6% to suppress background false positives.
4. **Validation & Test Preserved**: Validation (799 images) and test (307 images) distributions are **100% untouched** for unbiased evaluation.

---

## 💾 Caching Strategy in Google Colab

When running `IRC2026_YOLOv8n_Object_Detection.ipynb`, raw datasets are downloaded from Roboflow into high-speed local Colab SSD (`/content/raw_datasets`) and automatically archived as `.zip` files to Google Drive (`/content/drive/MyDrive/IRC2026_Rover_Vision/datasets/raw/`).

On subsequent Colab runs, the notebook unpacks the cached archives from Google Drive directly, saving bandwidth and avoiding Roboflow API limits.
