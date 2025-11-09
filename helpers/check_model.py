import torch
model = torch.load('models/pretrained/model_weights.pt', map_location='cpu')
print('Model keys:', list(model.keys()))
print('Sample weights shape:', model['backbone.layer4.2.conv3.weight'].shape if 'backbone.layer4.2.conv3.weight' in model else 'Key not found')
