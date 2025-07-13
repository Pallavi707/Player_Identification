# --- Cross-Camera Player Re-Identification using OSNet and Mutual Matching ---
from torchvision.models import resnet50
from scipy.optimize import linear_sum_assignment
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision import transforms
from torchreid.utils import FeatureExtractor
from scipy.optimize import linear_sum_assignment

# ------------------------- CONFIG -------------------------
# Paths for video and output directories
VIDEO_DIR = "Videos"
OUTPUT_DIR ="Results"


CONF_THRESHOLD = 0.7
SIMILARITY_THRESHOLD = 0.90
MAX_MOTION_LEN = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ MODELS --------------------------
yolo = YOLO("best.pt")
tracker = DeepSort(max_age=30, n_init=2)

# OSNet for person re-ID
extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path="osnet_x1_0_imagenet.pth",
    device=str(device)
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# -----------------------------------------------------------------------------------
# Function: extract_embedding
# Purpose:
#   Extracts an appearance-based embedding vector from a cropped image of a player
#   using a pre-trained OSNet model for person re-identification.
# Details:
#   - Converts the image to a tensor and normalizes it.
#   - Passes it through OSNet to extract a feature vector.
#   - Applies L2 normalization to the embedding for consistency.
#   - Returns None if the crop is too small or the process fails.
# Importance:
#   Provides visual features that help uniquely identify players regardless of camera view.
# -----------------------------------------------------------------------------------

def extract_embedding(crop):
    if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    try:
        img_tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = extractor(img_tensor)[0]
      
        emb = torch.nn.functional.normalize(emb[0].clone().detach(), dim=0).cpu().numpy()

        return emb
    except:
        return None
    
    
    
# -----------------------------------------------------------------------------------
# Function: extract_track_features
# Purpose:
#   Extracts and aggregates tracking features (appearance, position, motion) 
#   for each player across all frames in a video.
# Details:
#   - Runs YOLOv8 to detect players (class=2) in each frame.
#   - Tracks detected players using DeepSORT.
#   - For each track, extracts:
#       - Appearance embeddings from cropped player images.
#       - Normalized (x, y) position of the player in the frame.
#       - Recent motion trajectory (change in position over time).
#   - Aggregates these into one vector per player ID:
#       - Mean appearance embedding.
#       - Mean position.
#       - Flattened recent motion vector (max length = MAX_MOTION_LEN).
#   - Filters out invalid or short tracks.
# Importance:
#   Builds discriminative multi-modal player representations required for matching.
# -----------------------------------------------------------------------------------


def extract_track_features(video_path):
    cap = cv2.VideoCapture(video_path)
    track_data = {}
    frame_id = 0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 2 and conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0]
                detections.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], conf, "player"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx = (l + r) / 2 / frame_w
            cy = (t + b) / 2 / frame_h
            crop = frame[t:b, l:r]
            feature = extract_embedding(crop)
            if feature is None:
                continue
            if tid not in track_data:
                track_data[tid] = {"features": [], "positions": []}
            track_data[tid]["features"].append(feature)
            track_data[tid]["positions"].append([cx, cy])

        frame_id += 1

    cap.release()

    
    # Aggregate features
    final_tracks = {}
    for tid, data in track_data.items():
        feats = np.array(data["features"])
        pos = np.array(data["positions"])

        if len(feats) == 0 or len(pos) < 2:
            continue  # Skip empty or static tracks

        avg_feat = feats.mean(axis=0)
        norm = np.linalg.norm(avg_feat)
        if norm == 0 or np.isnan(norm):
            continue  # Skip corrupted or zero vectors
        avg_feat = avg_feat / norm

        avg_pos = pos.mean(axis=0)

        motion = pos[1:] - pos[:-1]
        motion_flat = np.zeros(MAX_MOTION_LEN * 2)
        valid_motion = motion[-MAX_MOTION_LEN:]
        motion_flat[:valid_motion.size] = valid_motion.flatten()

        final_tracks[tid] = (avg_feat, avg_pos, motion_flat)


    return final_tracks

  # -----------------------------------------------------------------------------------
# Function: bidirectional_match
# Purpose:
#   Matches players across two videos (e.g., broadcast and tacticam) using 
#   appearance, position, and motion features, with mutual consistency.
# Details:
#   - Uses Hungarian algorithm to find the best matchings in both directions:
#       - From source to target.
#       - From target to source.
#   - Matching score = weighted combination of:
#       - Cosine similarity of embeddings (appearance).
#       - Euclidean distance of average positions.
#       - Euclidean distance of motion vectors.
#   - Only retains mutual matches (player A in view1 matched to player B in view2,
#     and vice versa).
# Importance:
#   Provides robust and reliable player ID mapping between views, reducing false matches.
# -----------------------------------------------------------------------------------



def bidirectional_match(source, target):
    def hungarian_mapping(src, tgt):
        src_ids = list(src.keys())
        tgt_ids = list(tgt.keys())
        cost_matrix = np.zeros((len(tgt_ids), len(src_ids)))

        for i, tid in enumerate(tgt_ids):
            t_feat, t_pos, t_mov = tgt[tid]
            for j, sid in enumerate(src_ids):
                s_feat, s_pos, s_mov = src[sid]

                emb_score = cosine_similarity(t_feat.reshape(1, -1), s_feat.reshape(1, -1))[0][0]
                pos_score = 1 - np.linalg.norm(t_pos - s_pos)
                mov_score = 1 - np.linalg.norm(t_mov - s_mov)

                total_score = 0.6 * emb_score + 0.25 * pos_score + 0.15 * mov_score
                cost_matrix[i, j] = 1 - total_score

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = {}
        for i, j in zip(row_ind, col_ind):
            score = 1 - cost_matrix[i, j]
            if score >= SIMILARITY_THRESHOLD:
                mapping[tgt_ids[i]] = src_ids[j]
        return mapping

    fwd = hungarian_mapping(source, target)
    rev = hungarian_mapping(target, source)

    # Mutual match only
    final_map = {}
    for k, v in fwd.items():
        if rev.get(v) == k:
            final_map[k] = v
    return final_map

# -----------------------------------------------------------------------------------
# Function: remap_and_save
# Purpose:
#   Reruns detection and tracking on a video and overlays consistent remapped IDs
#   based on the provided `id_map`, then saves the annotated video.
# Details:
#   - Loads the video and detects players in each frame.
#   - Uses DeepSORT to track players.
#   - Uses `id_map` to replace local track IDs with globally matched ones.
#   - Draws bounding boxes and updated IDs on each frame.
#   - Writes the final annotated video to disk.
# Importance:
#   Enables visual verification of correct cross-camera identity assignments.
# -----------------------------------------------------------------------------------


def remap_and_save(video_path, id_map, output_path):
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(3)), int(cap.get(4))))
    tracker = DeepSort(max_age=30, n_init=2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 2 and conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0]
                detections.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], conf, "player"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid_int = int(track.track_id)
            mapped_id = id_map.get(tid_int, tid_int + 1000)
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {mapped_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()


# ---------------------------------MAIN --------------------------------------------

# Main Execution Block
# Purpose:
#   Orchestrates the full re-identification pipeline.
# Steps:
#   1. Loads broadcast and tacticam videos.
#   2. Extracts player features (appearance, motion, position) from both videos.
#   3. Performs mutual ID matching between the two views.
#   4. Annotates and saves the remapped tacticam video with consistent IDs.
#   5. Optionally saves a copy of the broadcast video with its tracked IDs.
# Importance:
#   This is the entry point for executing the cross-camera re-ID workflow.
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    broadcast_path = os.path.join(VIDEO_DIR, "broadcast.mp4")
    tacticam_path = os.path.join(VIDEO_DIR, "tacticam.mp4")

    print("Extracting features from broadcast.mp4...")
    b_tracks = extract_track_features(broadcast_path)

    print("Extracting features from tacticam.mp4...")
    t_tracks = extract_track_features(tacticam_path)

    print("Bidirectional mutual match...")
    id_map = bidirectional_match(b_tracks, t_tracks)

    print("Saving remapped tacticam.mp4...")
    remap_and_save(tacticam_path, id_map, os.path.join(OUTPUT_DIR, "tacticam_remapped.mp4"))

    print("Saving broadcast tracked.mp4...")
    remap_and_save(broadcast_path, {k: k for k in b_tracks}, os.path.join(OUTPUT_DIR, "broadcast_tracked.mp4"))

    print("Done.")
    
    