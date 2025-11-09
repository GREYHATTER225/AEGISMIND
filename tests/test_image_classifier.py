import torch
from models.image_classifier import ImageClassifier

model = ImageClassifier()
model.load_state_dict(torch.load('models/pretrained/image_classifier.pt', map_location='cpu'))
model.eval()

dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(dummy)
    print('Image classifier output shape:', out.shape)
    print('Sample output:', out.item())
    print('Is fake (>0.5):', out.item() > 0.5)
