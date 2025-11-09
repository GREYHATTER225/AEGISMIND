"""
AEGISMind Deepfake Model Audit — Local Debug Run
Objective: Analyze dataset integrity, model output logic, and threshold calibration.
"""

import torch
from models.image_classifier import ImageClassifier
from datasets.image_dataset import BalancedImageDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model with available checkpoint
model = ImageClassifier().to(device)

# Try different checkpoint paths
checkpoint_paths = [
    "models/pretrained/image_classifier.pt",
    "models/pretrained/image_classifier_checkpoint.pt", 
    "models/pretrained/model_weights.pt"
]

model_loaded = False
for path in checkpoint_paths:
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded model from: {path}")
            model_loaded = True
            break
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

if not model_loaded:
    print("No valid checkpoint found, using untrained model")
    
model.eval()

# 1️⃣ Check dataset label balance
dataset_path = "datasets/train"
real_count = len([f for f in os.listdir(os.path.join(dataset_path, "real_images")) if f.endswith(('.jpg', '.jpeg', '.png'))])
fake_count = len([f for f in os.listdir(os.path.join(dataset_path, "fake_images")) if f.endswith(('.jpg', '.jpeg', '.png'))])
print(f"Dataset balance: Real={real_count}, Fake={fake_count}")

# 2️⃣ Sample output check
try:
    dataset = BalancedImageDataset(root_dir=dataset_path, augment=False)
    print(f"Total dataset samples: {len(dataset)}")
    print(f"Class counts: {dataset.class_counts}")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    real_probs, fake_probs = [], []
    sample_count = 0
    max_samples = 100  # Limit samples for faster execution
    
    with torch.no_grad():
        for batch in loader:
            if sample_count >= max_samples:
                break
                
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            label = labels.item()
            
            if label == 0:  # real
                real_probs.append(probs[0])
            elif label == 1:  # fake
                fake_probs.append(probs[0])
            
            sample_count += 1
    
    if real_probs and fake_probs:
        real_avg = sum(real_probs)/len(real_probs)
        fake_avg = sum(fake_probs)/len(fake_probs)
        
        print(f"Real samples avg prob: {real_avg:.3f} (n={len(real_probs)})")
        print(f"Fake samples avg prob: {fake_avg:.3f} (n={len(fake_probs)})")
        
        # 3️⃣ Threshold check
        suggested_threshold = (real_avg + fake_avg) / 2
        print(f"Suggested decision threshold: {suggested_threshold:.3f}")
        
        # 4️⃣ Final recommendation
        separation = abs(real_avg - fake_avg)
        print(f"Class separation: {separation:.3f}")
        
        if separation < 0.3:
            print("Model confusion detected — retrain with more samples or verify labels.")
        else:
            print("Model separation looks solid — adjust threshold accordingly.")
            
        # Additional analysis
        print(f"\nDetailed Analysis:")
        print(f"   Real samples: {len(real_probs)} samples, avg prob: {real_avg:.3f}")
        print(f"   Fake samples: {len(fake_probs)} samples, avg prob: {fake_avg:.3f}")
        print(f"   Optimal threshold: {suggested_threshold:.3f}")
        print(f"   Separation quality: {'Good' if separation > 0.5 else 'Fair' if separation > 0.3 else 'Poor'}")
        
    else:
        print("No valid samples processed")
        
except Exception as e:
    print(f"Error during analysis: {e}")
    import traceback
    traceback.print_exc()
