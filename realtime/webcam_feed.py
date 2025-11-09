import cv2

def webcam_frame_generator(camera_index=0, max_frames=None):
    """
    Generator that yields (frame_index, frame) from webcam.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError(f"Could not open webcam at index {camera_index}")
    
    frame_idx = 0
    while max_frames is None or frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1
    
    cap.release()
