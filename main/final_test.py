"""
FINAL VERIFICATION TEST
Test the corrected inference logic on multiple samples
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from models.image_classifier import ImageClassifier
from datasets.image_dataset import BalancedImageDataset

def final_verification_test():
    print("FINAL VERIFICATION TEST")
    print("=" * 50)
    
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
    
    # Test function with robust threshold
    def predict_robust(image_tensor, model):
        with torch.no_grad():
            output = model(image_tensor)
            prob = torch.sigmoid(output).item()
            
            # ROBUST LOGIC: threshold 0.7
            if prob > 0.7:
                classification = "Real"
                confidence = prob
            else:
                classification = "Deepfake"
                confidence = 1 - prob
                
            return classification, confidence, prob
    
    # Test 5 real samples
    print("TESTING REAL SAMPLES:")
    real_samples = [s for s in dataset.samples if s['label'] == 0][:5]
    correct_real = 0
    
    for i, sample in enumerate(real_samples):
        try:
            image = Image.open(sample['path']).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            
            classification, confidence, prob = predict_robust(tensor, model)
            is_correct = classification == "Real"
            if is_correct:
                correct_real += 1
                
            print(f"  Real {i+1}: {prob:.4f} -> {classification} ({'CORRECT' if is_correct else 'WRONG'})")
        except Exception as e:
            print(f"  Error with real sample {i+1}: {e}")
    
    # Test 5 fake samples
    print("\nTESTING FAKE SAMPLES:")
    fake_samples = [s for s in dataset.samples if s['label'] == 1][:5]
    correct_fake = 0
    
    for i, sample in enumerate(fake_samples):
        try:
            image = Image.open(sample['path']).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            
            classification, confidence, prob = predict_robust(tensor, model)
            is_correct = classification == "Deepfake"
            if is_correct:
                correct_fake += 1
                
            print(f"  Fake {i+1}: {prob:.4f} -> {classification} ({'CORRECT' if is_correct else 'WRONG'})")
        except Exception as e:
            print(f"  Error with fake sample {i+1}: {e}")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"Real samples: {correct_real}/{len(real_samples)} correct ({correct_real/len(real_samples)*100:.1f}%)")
    print(f"Fake samples: {correct_fake}/{len(fake_samples)} correct ({correct_fake/len(fake_samples)*100:.1f}%)")
    print(f"Overall: {(correct_real + correct_fake)}/{len(real_samples) + len(fake_samples)} correct")
    
    if correct_real >= 4 and correct_fake >= 4:
        print("\nSUCCESS: Model is working correctly with robust threshold!")
    else:
        print("\nWARNING: Some samples still misclassified. Consider:")
        print("1. Check dataset labeling")
        print("2. Adjust threshold further")
        print("3. Retrain model")

if __name__ == "__main__":
    final_verification_test()
