
# CORRECTED INFERENCE LOGIC
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("path/to/model_weights.pt", map_location=device)
model.eval()

def predict_image_corrected(image_tensor, model):
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        
        # Based on audit: real samples give high prob (0.999), fake give low (0.013)
        # So: high prob = REAL, low prob = FAKE
        if prob > 0.5:  # High probability
            classification = "Real"
            confidence = prob
        else:  # Low probability  
            classification = "fake"
            confidence = 1 - prob
            
        print(f"Raw: {output.item():.4f} | Sigmoid: {prob:.4f} | Pred: {classification}")
        return classification, confidence
