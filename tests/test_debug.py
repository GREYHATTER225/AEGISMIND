import torch
from models.image_classifier import ImageClassifier
from datasets.image_dataset import BalancedImageDataset

model = ImageClassifier()
model_path = 'models/pretrained/image_classifier.pt'
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model.eval()

dataset = BalancedImageDataset('datasets/train')

# Test multiple samples
for i in [0, 100, 200, 300, 400, 500]:
    if i < len(dataset):
        sample = dataset[i]
        with torch.no_grad():
            out = model(sample['image'].unsqueeze(0))
            prob = torch.sigmoid(out).item()
            pred = 1 if prob > 0.5 else 0
            print(f'Sample {i}: Raw logit: {out.item():.4f}, Prob: {prob:.4f}, Pred: {pred}, True: {sample["label"].item()}')
