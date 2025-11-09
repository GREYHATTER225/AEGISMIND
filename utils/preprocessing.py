import torch
from PIL import Image
import torchvision.transforms as transforms

def pil_to_tensor(pil_image):
    """
    Convert PIL image to PyTorch tensor with normalization.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_image)

def spectrum_to_tensor(spectrum_image):
    """
    Convert spectrum image (numpy array) to PyTorch tensor.
    """
    # Assuming spectrum_image is a numpy array
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(spectrum_image)
