# Player Re-Identification Across Camera Views

## Overview

This project addresses the problem of player re-identification in soccer matches across two different camera views: `broadcast.mp4` and `tacticam.mp4`. The goal is to assign consistent IDs to players in both videos by leveraging deep visual features, spatial positions, and motion patterns.

We use a combination of YOLOv11 (for player detection), DeepSORT (for short-term tracking), and OSNet (for person re-identification via visual embeddings). After extracting features from both videos, we use bidirectional matching to map tacticam player IDs to broadcast player IDs.

---

## Folder Structure

```
Project Folder/
├── Main.ipynb                 # Main script for running the pipeline
├── README.md               # This file
├── Assignment Materials/
│   ├── broadcast.mp4       # Broadcast view of the game
│   └── tacticam.mp4        # Tacticam view of the game
├── Results/
│   ├── broadcast_tracked.mp4
│   └── tacticam_remapped.mp4
├── Player_ID_Detection.ipynb # For cheking the result of best.pt model
```

---

## Setup Instructions

### 1. Install Dependencies

Ensure you're using Python 3.8+ and run:

```bash
pip install -r requirements.txt
```

**Required libraries:**

* ultralytics
* torch, torchvision
* opencv-python
* deep\_sort\_realtime
* scikit-learn
* torchreid

### 2. Download Detection Model

Download the provided YOLOv11 fine-tuned model and place it as:

```
best.pt  -->  /content/drive/MyDrive/Stealth_Mode/best.pt
```

### 3. Download OSNet Pretrained Weights

Place the file `osnet_x1_0_imagenet.pth` inside the root directory or `~/.cache/torch/checkpoints/`.
You can download it from:

```
https://github.com/KaiyangZhou/deep-person-reid/releases
```

---

## How to Run

```bash
python main.py
```

This will:

1. Detect and track players in both videos.
2. Extract average appearance, position, and motion features.
3. Match tacticam tracks to broadcast tracks using bidirectional Hungarian matching.
4. Save two new videos:

   * `broadcast_tracked.mp4` with original IDs.
   * `tacticam_remapped.mp4` with IDs aligned to broadcast.

---

## Features Used

* **Visual Appearance**: Extracted using OSNet from cropped player images.
* **Spatial Location**: Normalized player positions within the frame.
* **Motion Vectors**: Displacement patterns over 10 most recent positions.

These features are combined into a matching score:

```
score = 0.6 * appearance_similarity + 0.25 * position_similarity + 0.15 * motion_similarity
```
---

## Authors

Pallavi Singh - MSc AI & Sustainable Development - UCL


---

## License

MIT License. Provided for academic and non-commercial use only.
