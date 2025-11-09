import cv2
import numpy as np
import tempfile
import os

def frame_generator_from_path_or_fileobj(video_path_or_fileobj, max_frames=120):
    """
    Generator that yields (frame_index, frame) from a video file path or file-like object.
    """
    temp_file_path = None
    if hasattr(video_path_or_fileobj, 'read'):
        # It's a file-like object, save to temporary file
        video_data = video_path_or_fileobj.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_data)
            temp_file_path = temp_file.name
        cap = cv2.VideoCapture(temp_file_path)
    else:
        # It's a path
        cap = cv2.VideoCapture(video_path_or_fileobj)
    
    if not cap.isOpened():
        if temp_file_path:
            os.unlink(temp_file_path)
        raise ValueError("Could not open video file")
    
    frame_idx = 0
    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()
        if temp_file_path:
            os.unlink(temp_file_path)
