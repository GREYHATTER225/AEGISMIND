"""
Detailed Sample Analysis - Check Multiple Samples
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from models.image_classifier import ImageClassifier
from datasets.image_dataset import BalancedImageDataset

def analyze_multiple_samples():
    print("Analyzing Multiple Samples...")
    
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
    
    # Test 5 real samples
    print("=== REAL SAMPLES ===")
    real_samples = [s for s in dataset.samples if s['label'] == 0][:5]
    for i, sample in enumerate(real_samples):
        try:
            image = Image.open(sample['path']).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(tensor)
                prob = torch.sigmoid(output).item()
                
                if prob > 0.5:
                    classification = "Real"
                else:
                    classification = "Deepfake"
                    
            print(f"Real {i+1}: {prob:.4f} -> {classification} ({sample['path'][-20:]})")
        except Exception as e:
            print(f"Error with real sample {i+1}: {e}")
    
    # Test 5 fake samples
    print("\n=== FAKE SAMPLES ===")
    fake_samples = [s for s in dataset.samples if s['label'] == 1][:5]
    for i, sample in enumerate(fake_samples):
        try:
            image = Image.open(sample['path']).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(tensor)
                prob = torch.sigmoid(output).item()
                
                if prob > 0.5:
                    classification = "Real"
                else:
                    classification = "Deepfake"
                    
            print(f"Fake {i+1}: {prob:.4f} -> {classification} ({sample['path'][-20:]})")
        except Exception as e:
            print(f"Error with fake sample {i+1}: {e}")
    
    # Calculate averages
    print("\n=== STATISTICS ===")
    real_probs = []
    fake_probs = []
    
    for sample in dataset.samples[:50]:  # Test first 50 samples
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
        except:
            continue
    
    if real_probs and fake_probs:
        real_avg = sum(real_probs) / len(real_probs)
        fake_avg = sum(fake_probs) / len(fake_probs)
        
        print(f"Real samples (n={len(real_probs)}): avg={real_avg:.4f}")
        print(f"Fake samples (n={len(fake_probs)}): avg={fake_avg:.4f}")
        print(f"Separation: {abs(real_avg - fake_avg):.4f}")
        
        # Determine optimal threshold
        optimal_threshold = (real_avg + fake_avg) / 2
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        
        # Test optimal threshold
        correct_real = sum(1 for p in real_probs if p > optimal_threshold)
        correct_fake = sum(1 for p in fake_probs if p <= optimal_threshold)
        
        print(f"With optimal threshold:")
        print(f"  Real accuracy: {correct_real}/{len(real_probs)} = {correct_real/len(real_probs)*100:.1f}%")
        print(f"  Fake accuracy: {correct_fake}/{len(fake_probs)} = {correct_fake/len(fake_probs)*100:.1f}%")

if __name__ == "__main__":
    analyze_multiple_samples()
