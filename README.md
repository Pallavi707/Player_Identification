# Player Re-Identification in Sports Footage (Liat.ai Assignment)

## Objective

This project solves the **Cross-Camera Player Mapping** problem for soccer footage. Given two videos — `broadcast.mp4` and `tacticam.mp4` — showing the same match from different camera angles, the goal is to ensure **players retain consistent IDs** across views.

We use a **YOLOv11 detector** combined with **ResNet-18-based appearance embeddings**, **spatial location**, and **motion features** to compute cross-camera identity mappings.
