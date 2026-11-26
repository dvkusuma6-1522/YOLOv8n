# Dataset 1: Cone Detector

- **Roboflow Workspace**: `21-jyqut`
- **Roboflow Project**: `cone-detector-uexft`
- **Version**: `1`
- **Universe URL**: [https://universe.roboflow.com/21-jyqut/cone-detector-uexft](https://universe.roboflow.com/21-jyqut/cone-detector-uexft)
- **License**: CC BY 4.0
- **Declared Classes**: `['cone detection - v1 2024-08-13 3-16pm']` (nc: 1)
- **Unified Target Mapping**: Class ID `0` $\rightarrow$ Target Class `0: cone`

### Characteristics & Audit Notes
- **Split Counts**: Train (5,862 images), Valid (558 images), Test (200 images).
- **Annotation Format**: Primarily standard YOLO bounding boxes with some Roboflow segmentation polygons ($>5$ tokens).
- **Processing in Pipeline**: Polygons are automatically converted to tight bounding box envelopes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \rightarrow [x_c, y_c, w, h]$, and coordinates are clipped to $[0.0, 1.0]$. Pure cone images are subsampled during training to prevent class imbalance.
