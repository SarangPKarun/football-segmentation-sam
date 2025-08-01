import cv2
import os
from PIL import Image

def extract_frames_from_video(video_path, output_folder, every_n=1):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")

    saved_count = 0
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % every_n == 0:
            filename = f"{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)
            saved_count += 1

        idx += 1

    cap.release()
    print(f"✅ Saved {saved_count} frames every {every_n} frames.")


def images_to_video(input_folder, output_path, fps=25):
    # ✅ Sort images by number extracted from filename
    images = sorted([
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']
    ], key=lambda x: int(os.path.splitext(x)[0].replace("seg_", "")))

    if not images:
        raise ValueError("No valid images found in the folder")

    # ✅ Get frame size from the first image
    first_image_path = os.path.join(input_folder, images[0])
    with Image.open(first_image_path) as img:
        width, height = img.size

    # ✅ Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_name in images:
        image_path = os.path.join(input_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"⚠️ Skipping unreadable image: {image_path}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"✅ Video saved to: {output_path}")