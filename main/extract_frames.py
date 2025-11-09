import cv2
import os
from PIL import Image

def extract_first_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return False

    ret, frame = cap.read()
    if ret:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.save(output_path)
        print(f"Saved frame to {output_path}")
        cap.release()
        return True
    else:
        print(f"Cannot read frame from {video_path}")
        cap.release()
        return False

import argparse

def main(data_dir='datasets/train'):
    # Extract one frame from each video for training
    os.makedirs(f'{data_dir}/real_images', exist_ok=True)
    os.makedirs(f'{data_dir}/fake_images', exist_ok=True)

    real_videos = os.listdir(f'{data_dir}/real')
    fake_videos = os.listdir(f'{data_dir}/fake')

    print(f"Extracting frames from {len(real_videos)} real videos...")
    for i, video in enumerate(real_videos):
        if video.endswith('.mp4'):
            video_path = f'{data_dir}/real/{video}'
            output_path = f'{data_dir}/real_images/{video.replace(".mp4", ".jpg")}'
            extract_first_frame(video_path, output_path)

    print(f"Extracting frames from {len(fake_videos)} fake videos...")
    for i, video in enumerate(fake_videos):
        if video.endswith('.mp4'):
            video_path = f'{data_dir}/fake/{video}'
            output_path = f'{data_dir}/fake_images/{video.replace(".mp4", ".jpg")}'
            extract_first_frame(video_path, output_path)

    print("Frame extraction completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/train')
    args = parser.parse_args()
    main(args.data_dir)
