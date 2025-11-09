import torch
from models.image_classifier import ImageClassifier

model = ImageClassifier()
model_path = 'models/pretrained/image_classifier.pt'
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model.eval()

dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(dummy)
    print('Trained model output:', out.item())
    print('Is fake (>0.5):', out.item() > 0.5)
