import torch
from models.image_classifier import ImageClassifier
from datasets.image_dataset import BalancedImageDataset

model = ImageClassifier()
model_path = 'models/pretrained/image_classifier.pt'
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model.eval()

dataset = BalancedImageDataset('datasets/train')
sample = dataset[1]  # Get a fake sample

with torch.no_grad():
    out = model(sample['image'].unsqueeze(0))
    print('Trained model on fake sample:', out.item())
    print('Label:', sample['label'].item())
