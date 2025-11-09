import torch
from models.image_classifier import ImageClassifier
from datasets.image_dataset import BalancedImageDataset

model = ImageClassifier()
dataset = BalancedImageDataset('datasets/train')
sample = dataset[0]

model.eval()
with torch.no_grad():
    out = model(sample['image'].unsqueeze(0))
    print('Inference successful')
    print('Prediction:', out.item())
    print('Label:', sample['label'].item())
