import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt



sys.path.append("/home/sunya/Desktop/Footb/sam2")
from sam2.build_sam import build_sam2_video_predictor

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Build predictor
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
print(BASE_DIR)
checkpoint = os.path.join(BASE_DIR, "sam2", "checkpoints", "sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


# Video directory & frame listing
video_dir = "static/frames"  # or "../football" if you're working outside Flask static
frame_names = sorted([
    f for f in os.listdir(video_dir)
    if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg"]
], key=lambda p: int(os.path.splitext(p)[0]))

# Mask and point overlay functions
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


# inference_state = predictor.init_state(video_path=video_dir)
# predictor.reset_state(inference_state)

def initialize_inference_state(video_dir):
    # predictor.reset()  # optional if you reuse predictor
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    return predictor, inference_state