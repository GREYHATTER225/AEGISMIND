import torch
from models.resnext_lstm import DeepfakeDetector
from datasets.deepfake_dataset import DeepfakeDataset
from torch.utils.data import DataLoader

# Load model
model = DeepfakeDetector()
model.load_state_dict(torch.load('models/pretrained/model_weights.pt', map_location='cpu'))
model.eval()

# Load a few samples
dataset = DeepfakeDataset('datasets/train')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Test predictions
with torch.no_grad():
    for batch in dataloader:
        frames = batch['frames']
        labels = batch['label']
        outputs = model(frames)
        preds = (outputs > 0.5).float()

        print("Labels:", labels.numpy())
        print("Predictions:", preds.numpy())
        print("Outputs:", outputs.numpy())
        break
