import cv2
from PIL import Image
import os
import numpy as np

def extract_and_preprocess(video_path, frame_rate=1, num_frames=6):
    """
    Extracts frames from a video (or single image) and returns them as numpy arrays.

    Args:
        video_path (str): Path to video file or image.
        frame_rate (int): Number of frames per second to extract (default 1).
        num_frames (int): Number of frames to extract (default 6).

    Returns:
        List[np.ndarray]: List of frames as numpy arrays [H, W, C].
    """

    frames = []

    # Check if input is an image
    if os.path.splitext(video_path)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        try:
            img = Image.open(video_path).convert('RGB')
            img = img.resize((224, 224))
            frame = np.array(img)
            frames.append(frame)
        except Exception as e:
            print(f"Error processing image {video_path}: {e}")
        return frames

    # Else treat as video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return frames

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / frame_rate) if video_fps > 0 else 1

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            # Convert BGR to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((224, 224))
            frame_resized = np.array(img)
            frames.append(frame_resized)

            # Stop if we have enough frames
            if len(frames) >= num_frames:
                break

        count += 1

    cap.release()
    return frames

# Example usage
if __name__ == "__main__":
    sample_path = "datasets/samples/sample_video.mp4"
    frames = extract_and_preprocess(sample_path, frame_rate=1)
    print(f"Extracted {len(frames)} frames.")
