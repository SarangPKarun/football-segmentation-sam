from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
from utils.video_utils import extract_frames_from_video
from utils.segment_runner import run_segmentation_from_clicks
import numpy as np


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
FRAME_FOLDER = 'static/frames'
CLICK_OUTPUT = 'clicks.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video_file = request.files['video']
    if video_file.filename == '':
        return 'No selected file'

    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    # Clear existing frames
    for f in os.listdir(FRAME_FOLDER):
        os.remove(os.path.join(FRAME_FOLDER, f))

    # Extract frames
    extract_frames_from_video(video_path, FRAME_FOLDER, every_n=1)

    return redirect(url_for('annotate'))

@app.route('/annotate')
def annotate():
    frame_names = sorted(os.listdir(FRAME_FOLDER))

    total = len(frame_names)
    selected_frames = np.linspace(0, total - 1, 10, dtype=int)
    selected_frames = [frame_names[i] for i in selected_frames]



    previous_clicks = {}
    try:
        if os.path.exists(CLICK_OUTPUT) and os.path.getsize(CLICK_OUTPUT) > 0:
            with open(CLICK_OUTPUT, "r") as f:
                previous_clicks = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        previous_clicks = {}


    return render_template("index.html", frame_names=selected_frames, previous_clicks=previous_clicks)



@app.route('/submit_click', methods=['POST'])
def submit_click():
    data = request.get_json()
    frame = data['frame']
    x = data['x']
    y = data['y']
    label = data['label']
    print(data)

    clicks = {}
    if os.path.exists(CLICK_OUTPUT):
        with open(CLICK_OUTPUT, 'r') as f:
            clicks = json.load(f)

    # --- 🗑️ Delete frame entirely if x, y, label are all None ---
    if x is None and y is None and label is None:
        if frame in clicks:
            del clicks[frame]
            print(f"Deleted all clicks from frame: {frame}")
    # --- ✅ Add a new point ---
    elif frame and x is not None and y is not None and label:
        if frame not in clicks:
            clicks[frame] = {}

        if label not in clicks[frame]:
            clicks[frame][label] = []

        clicks[frame][label].append([x, y])
        print(f"Added ({x}, {y}) to {label} in frame {frame}")

    with open(CLICK_OUTPUT, 'w') as f:
        json.dump(clicks, f, indent=2)

    return jsonify({"frame": frame, "x": x, "y": y, "label": label})



@app.route('/segment', methods=['POST'])
def segment_ball():
    print("Segment endpoint hit")  
    success, message = run_segmentation_from_clicks()
    return jsonify({"success": success, "message": message})

@app.route('/clear_clicks', methods=['POST'])
def clear_clicks():
    try:
        with open(CLICK_OUTPUT, 'w') as f:
            json.dump({}, f)
        return jsonify({"message": "✅ All clicks cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
