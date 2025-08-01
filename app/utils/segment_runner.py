import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
from PIL import Image
import json
from flask import jsonify
from utils.video_utils import images_to_video



from utils.sam_setup import initialize_inference_state, video_dir, frame_names, show_points, show_mask  # adjust as per your structure

def run_segmentation_from_clicks(click_file='clicks.json', output_img_path='static/seg_frame'):

    # Video directory & frame listing
    video_dir = "static/frames"  # or "../football" if you're working outside Flask static

    # ✅ Check if frames exist
    if not os.path.exists(video_dir) or not os.listdir(video_dir):
        return jsonify(error="⚠️ No frames found. Please upload a video first."), 400
    
    # Remove all existing files in the output folder
    if os.path.exists(output_img_path):
        for f in os.listdir(output_img_path):
            file_path = os.path.join(output_img_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_img_path, exist_ok=True)

    
    frame_names = sorted([
        f for f in os.listdir(video_dir)
        if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg"]
    ], key=lambda p: int(os.path.splitext(p)[0]))

    # ✅ Only initialize after frames are ready
    print("Loading frames...")
    predictor, inference_state = initialize_inference_state(video_dir)
    print("Frames loaded.")

    try:
        with open(click_file, "r") as f:
            click_data = json.load(f)
            print(click_data)
    except FileNotFoundError:
        return "No click data found."

    ann_obj_id = 1
    for frame_name, coords in click_data.items():
        ann_frame_idx = frame_names.index(frame_name)
        points = np.array([coords], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        predictor.reset_state(inference_state)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

    # --- Step 1: Run SAM2 propagation on full video ---
    video_segments = {}  # holds masks per frame
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # --- Step 2: Visualize and save segmentation for every N frames ---
    vis_frame_stride = 1  # You can set this to 1 if you want every frame
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        frame_name = frame_names[out_frame_idx]
        frame_path = os.path.join(video_dir, frame_name)
        image = Image.open(frame_path)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")

        for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
            show_mask(out_mask, ax, obj_id=out_obj_id)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        output_path = os.path.join(output_img_path, f"seg_{frame_name}")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    images_to_video(output_img_path, 'static/segmentation_output.mp4')


    
    

    return True, "Per-frame segmentation completed."

