# Dataset 3: White Chalk Circle and Cone

- **Roboflow Workspace**: `kusuma-dv`
- **Roboflow Project**: `white-chalk-circle-and-cone`
- **Version**: `3`
- **Universe URL**: [https://universe.roboflow.com/kusuma-dv/white-chalk-circle-and-cone](https://universe.roboflow.com/kusuma-dv/white-chalk-circle-and-cone)
- **License**: CC BY 4.0
- **Declared Classes**: `['cone', 'white_chalk_circle']` (nc: 2)
- **Unified Target Mapping**:
  - `0: cone` $\rightarrow$ Target Class `0: cone`
  - `1: white_chalk_circle` $\rightarrow$ Target Class `1: circle`

### Characteristics & Audit Notes
- **Split Counts**: Train (958 images), Valid (35 images), Test (20 images).
- **Purpose**: Direct competition-domain imagery featuring white chalk circles drawn on gravel, dirt, asphalt, and natural outdoor rover terrain, frequently co-occurring with cones in the same frame.
- **Processing in Pipeline**: Preserves multi-object co-occurrence annotations; oversampled in the training split to enhance detection recall on faint chalk markings.
