# Dataset 2: Circle Dataset

- **Roboflow Workspace**: `deneme-3usbo`
- **Roboflow Project**: `circle-9po6m`
- **Version**: `1`
- **Universe URL**: [https://universe.roboflow.com/deneme-3usbo/circle-9po6m](https://universe.roboflow.com/deneme-3usbo/circle-9po6m)
- **License**: CC BY 4.0
- **Declared Classes**: `['Circle', 'blue circle', 'green circle', 'redcircle']` (nc: 4)
- **Unified Target Mapping**:
  - `0: Circle` $\rightarrow$ Target Class `1: circle`
  - `1: blue circle` $\rightarrow$ Target Class `1: circle`
  - `2: green circle` $\rightarrow$ Target Class `1: circle`
  - `3: redcircle` $\rightarrow$ Target Class `1: circle`

### Characteristics & Audit Notes
- **Split Counts**: Train (1,878 images), Valid (206 images), Test (87 images).
- **Purpose**: Provides geometrical circle features across multiple colors, viewpoints, and scales.
- **Processing in Pipeline**: All 4 circle subclasses are harmonized into a single unified `circle` class (Class 1). Out-of-bounds coordinates are clipped to $[0.0, 1.0]$.
