"""
FINAL SOLUTION: Robust Deepfake Detection Logic
This implements a more robust threshold that handles edge cases
"""

def predict_image_robust(image_tensor, model):
    """
    Robust prediction logic that handles edge cases
    """
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        
        # Use a more conservative threshold to handle edge cases
        # Based on our analysis: real avg=0.9925, fake avg=0.0111
        # Use 0.7 as threshold to be more conservative
        threshold = 0.7
        
        if prob > threshold:
            classification = "Real"
            confidence = prob
        else:
            classification = "Deepfake"
            confidence = 1 - prob
            
        print(f"Raw: {output.item():.4f} | Sigmoid: {prob:.4f} | Threshold: {threshold:.4f} | Pred: {classification}")
        return classification, confidence

def predict_video_frames_robust(frames, model, max_frames=30):
    """
    Robust video prediction logic
    """
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    frame_tensors = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        if frame.mode == 'RGBA':
            frame = frame.convert('RGB')
        tensor = preprocess_image(frame)
        frame_tensors.append(tensor.squeeze(0))
    
    video_tensor = torch.stack(frame_tensors)
    video_tensor = video_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(video_tensor)
        prob = torch.sigmoid(output).item()
        
        # Use conservative threshold
        threshold = 0.7
        
        if prob > threshold:
            classification = "Real"
            confidence = prob
        else:
            classification = "Deepfake"
            confidence = 1 - prob
            
        return classification, confidence, len(frames)

# Test the robust logic
if __name__ == "__main__":
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    from models.image_classifier import ImageClassifier
    from datasets.image_dataset import BalancedImageDataset
    
    print("Testing Robust Logic with Threshold 0.7...")
    
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
    
    # Test multiple samples
    print("\n=== TESTING ROBUST LOGIC ===")
    
    # Test 3 real samples
    real_samples = [s for s in dataset.samples if s['label'] == 0][:3]
    for i, sample in enumerate(real_samples):
        try:
            image = Image.open(sample['path']).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            
            classification, confidence = predict_image_robust(tensor, model)
            print(f"Real {i+1}: {classification} (conf: {confidence:.4f})")
        except Exception as e:
            print(f"Error with real sample {i+1}: {e}")
    
    # Test 3 fake samples
    fake_samples = [s for s in dataset.samples if s['label'] == 1][:3]
    for i, sample in enumerate(fake_samples):
        try:
            image = Image.open(sample['path']).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            
            classification, confidence = predict_image_robust(tensor, model)
            print(f"Fake {i+1}: {classification} (conf: {confidence:.4f})")
        except Exception as e:
            print(f"Error with fake sample {i+1}: {e}")
    
    print("\n=== SUMMARY ===")
    print("Robust threshold 0.7 should handle edge cases better")
    print("Real samples should mostly be classified as Real")
    print("Fake samples should mostly be classified as Deepfake")
