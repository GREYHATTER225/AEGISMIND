import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import random
import json
from tqdm import tqdm

def blend_face_onto_background(bg_path, face_path, out_path, bbox, scale=1.0, feather=10, rotation=0):
    """
    Blend a face onto a background image at specified bbox location.
    """
    bg = Image.open(bg_path).convert('RGB')
    face = Image.open(face_path).convert('RGBA')

    # Apply rotation if specified
    if rotation != 0:
        face = face.rotate(rotation, expand=True)

    # Scale face
    w, h = face.size
    new_w, new_h = int(w * scale), int(h * scale)
    face = face.resize((new_w, new_h), Image.LANCZOS)

    # Position based on bbox
    x, y, w_box, h_box = bbox
    # Center the face in the bbox
    x_pos = x + (w_box - new_w) // 2
    y_pos = y + (h_box - new_h) // 2

    # Ensure within bounds
    x_pos = max(0, min(x_pos, bg.width - new_w))
    y_pos = max(0, min(y_pos, bg.height - new_h))

    # Create mask with feathering
    mask = face.split()[3].filter(ImageFilter.GaussianBlur(feather))

    # Paste with alpha composite
    bg.paste(face, (x_pos, y_pos), mask)
    bg.save(out_path)

    # Return actual bbox used
    return [x_pos, y_pos, x_pos + new_w, y_pos + new_h]

def generate_composites(bg_dir, face_dir, output_dir, num_samples=1000, annotations_file='annotations.json'):
    """
    Generate composite images by pasting AI faces onto real backgrounds.
    """
    os.makedirs(output_dir, exist_ok=True)

    bg_files = [f for f in os.listdir(bg_dir) if f.endswith(('.jpg', '.png'))]
    face_files = [f for f in os.listdir(face_dir) if f.endswith(('.jpg', '.png'))]

    annotations = []

    for i in tqdm(range(num_samples), desc="Generating composites"):
        bg_file = random.choice(bg_files)
        face_file = random.choice(face_files)

        bg_path = os.path.join(bg_dir, bg_file)
        face_path = os.path.join(face_dir, face_file)

        # Random bbox (simulate where a person might be)
        bg_img = Image.open(bg_path)
        w, h = bg_img.size
        bbox_w = random.randint(50, min(200, w//2))
        bbox_h = random.randint(50, min(200, h//2))
        x = random.randint(0, w - bbox_w)
        y = random.randint(0, h - bbox_h)
        bbox = [x, y, bbox_w, bbox_h]

        # Random augmentations
        scale = random.uniform(0.8, 1.2)
        rotation = random.uniform(-15, 15)
        feather = random.randint(5, 15)

        out_path = os.path.join(output_dir, f"composite_{i:04d}.jpg")
        actual_bbox = blend_face_onto_background(bg_path, face_path, out_path, bbox, scale, feather, rotation)

        # Annotation
        ann = {
            "image_id": i,
            "file_name": f"composite_{i:04d}.jpg",
            "bbox": actual_bbox,  # [x1, y1, x2, y2]
            "category_id": 1,  # 1 for ai_added_person
            "area": (actual_bbox[2] - actual_bbox[0]) * (actual_bbox[3] - actual_bbox[1]),
            "iscrowd": 0
        }
        annotations.append(ann)

    # Save COCO-style annotations
    coco_ann = {
        "images": [{"id": ann["image_id"], "file_name": ann["file_name"], "width": 0, "height": 0} for ann in annotations],
        "annotations": annotations,
        "categories": [{"id": 0, "name": "real_person"}, {"id": 1, "name": "ai_added_person"}]
    }

    with open(annotations_file, 'w') as f:
        json.dump(coco_ann, f, indent=2)

    print(f"Generated {num_samples} composites. Annotations saved to {annotations_file}")

if __name__ == "__main__":
    # Example usage - adjust paths as needed
    bg_dir = "datasets/backgrounds"  # Directory with real scene images
    face_dir = "datasets/ai_faces"    # Directory with AI-generated faces
    output_dir = "datasets/composites"
    generate_composites(bg_dir, face_dir, output_dir, num_samples=500)
