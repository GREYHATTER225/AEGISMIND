import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import cv2
import numpy as np
from PIL import Image
import os

class FaceClassifier:
    def __init__(self, model_path='weights/face_classifier.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict_face(self, face_crop):
        """
        Predict if a face crop is real or AI-generated.
        face_crop: PIL Image or numpy array
        Returns: (prediction, confidence) where prediction is 0=real, 1=fake
        """
        if isinstance(face_crop, np.ndarray):
            face_crop = Image.fromarray(face_crop)

        face_crop = face_crop.convert('RGB')
        tensor = self.transform(face_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            prob = torch.sigmoid(outputs).item()  # Apply sigmoid for binary classification
            # ROBUST LOGIC: Use threshold 0.7 to handle edge cases
            pred = 0 if prob > 0.7 else 1  # 0=real, 1=fake
            confidence = prob if pred == 0 else 1 - prob  # Confidence for the predicted class

            # Debug prints
            print(f"Raw logit: {outputs.item():.4f}")
            print(f"Sigmoid probability: {prob:.4f}")
            print(f"Final predicted label: {pred} (0=real, 1=fake)")

        return pred, confidence

class GradCAMExplainer:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.feature_maps = None

        # Hook the gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.feature_maps = output

        # Assuming ResNet, hook the last conv layer
        self.model.layer4[-1].register_forward_hook(forward_hook)
        self.model.layer4[-1].register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()

        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=[0, 2, 3])
        cam = torch.zeros(self.feature_maps.shape[2:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * self.feature_maps[0, i, :, :]

        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam.cpu().numpy()

    def overlay_heatmap(self, image, heatmap, alpha=0.6):
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay

def detect_faces_and_classify(image, classifier, explainer, threshold=0.5):
    """
    Detect faces in image and classify each as real/AI.
    For now, using simple face detection. In full implementation, use proper detector.
    """
    # Placeholder: assume face bbox is provided or use simple detection
    # In real implementation, integrate with face detector like MTCNN or RetinaFace

    results = []

    # For demo, assume single face in center (replace with actual detection)
    h, w = image.shape[:2]
    face_bbox = [w//4, h//4, 3*w//4, 3*h//4]  # Placeholder bbox

    # Crop face
    x1, y1, x2, y2 = face_bbox
    face_crop = image[y1:y2, x1:x2]

    if face_crop.size > 0:
        pred, conf = classifier.predict_face(face_crop)

        # Generate explanation
        face_tensor = classifier.transform(Image.fromarray(face_crop)).unsqueeze(0).to(classifier.device)
        heatmap = explainer.generate_heatmap(face_tensor, target_class=pred)

        results.append({
            'bbox': face_bbox,
            'prediction': pred,  # 0=real, 1=ai
            'confidence': conf,
            'heatmap': heatmap
        })

    return results

# Example usage
if __name__ == "__main__":
    classifier = FaceClassifier()
    explainer = GradCAMExplainer(classifier.model)

    # Test on image
    image = cv2.imread('test_image.jpg')
    results = detect_faces_and_classify(image, classifier, explainer)

    for result in results:
        print(f"Face at {result['bbox']}: {'AI' if result['prediction'] else 'Real'} ({result['confidence']:.2f})")
