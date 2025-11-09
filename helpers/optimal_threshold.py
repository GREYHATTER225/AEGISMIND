"""
FINAL CORRECTED INFERENCE LOGIC
This uses a data-driven threshold based on actual model performance
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from models.image_classifier import ImageClassifier
from datasets.image_dataset import BalancedImageDataset

def calculate_optimal_threshold():
    """Calculate the optimal threshold based on actual data distribution"""
    print("Calculating optimal threshold...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier().to(device)
    
    # Load model
    model_path = "models/pretrained/image_classifier.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = BalancedImageDataset(root_dir="datasets/train", augment=False)
    
    real_probs = []
    fake_probs = []
    
    # Sample more data for better statistics
    sample_count = 0
    for sample in dataset.samples:
        if sample_count >= 200:  # Limit for speed
            break
        try:
            image = Image.open(sample['path']).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(tensor)
                prob = torch.sigmoid(output).item()
                
                if sample['label'] == 0:  # real
                    real_probs.append(prob)
                else:  # fake
                    fake_probs.append(prob)
                    
            sample_count += 1
        except:
            continue
    
    if real_probs and fake_probs:
        real_avg = sum(real_probs) / len(real_probs)
        fake_avg = sum(fake_probs) / len(fake_probs)
        
        # Calculate optimal threshold as midpoint
        optimal_threshold = (real_avg + fake_avg) / 2
        
        print(f"Real samples (n={len(real_probs)}): avg={real_avg:.4f}")
        print(f"Fake samples (n={len(fake_probs)}): avg={fake_avg:.4f}")
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        
        return optimal_threshold
    else:
        print("Using default threshold: 0.5")
        return 0.5

def predict_with_optimal_threshold(image_tensor, model, threshold=None):
    """Predict using optimal threshold"""
    if threshold is None:
        threshold = calculate_optimal_threshold()
    
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        
        # Use optimal threshold
        if prob > threshold:
            classification = "Real"
            confidence = prob
        else:
            classification = "Deepfake"
            confidence = 1 - prob
            
        print(f"Raw: {output.item():.4f} | Sigmoid: {prob:.4f} | Threshold: {threshold:.4f} | Pred: {classification}")
        return classification, confidence, threshold

# Test the corrected logic
if __name__ == "__main__":
    print("Testing Optimal Threshold Logic...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier().to(device)
    
    # Load model
    model_path = "models/pretrained/image_classifier.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Calculate optimal threshold
    optimal_threshold = calculate_optimal_threshold()
    
    # Test on samples
    dataset = BalancedImageDataset(root_dir="datasets/train", augment=False)
    
    print(f"\nTesting with optimal threshold: {optimal_threshold:.4f}")
    
    # Test real sample
    real_sample = None
    for sample in dataset.samples:
        if sample['label'] == 0:  # real
            real_sample = sample
            break
    
    if real_sample:
        print(f"\nReal sample test:")
        image = Image.open(real_sample['path']).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        classification, confidence, _ = predict_with_optimal_threshold(tensor, model, optimal_threshold)
    
    # Test fake sample
    fake_sample = None
    for sample in dataset.samples:
        if sample['label'] == 1:  # fake
            fake_sample = sample
            break
    
    if fake_sample:
        print(f"\nFake sample test:")
        image = Image.open(fake_sample['path']).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        classification, confidence, _ = predict_with_optimal_threshold(tensor, model, optimal_threshold)
