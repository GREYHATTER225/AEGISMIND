import torch
import torch.nn as nn
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import shap
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import librosa

class SingleFrameModel(nn.Module):
    def __init__(self, full_model):
        super(SingleFrameModel, self).__init__()
        # Check if model has LSTM (ResNeXt-LSTM) or not (ImageClassifier)
        self.has_lstm = hasattr(full_model, 'lstm')
        if self.has_lstm:
            # For ResNeXt-LSTM: backbone is ResNeXt without fc, classifier is separate
            self.backbone = full_model.backbone
            self.lstm = full_model.lstm
            # Use the classifier from the full model (remove sigmoid for logits)
            self.classifier = nn.Sequential(*list(full_model.classifier.children())[:-1])  # Remove sigmoid
        else:
            # For ImageClassifier: backbone is ResNet with fc replaced
            # We need to get features before the fc layer
            self.backbone = nn.Sequential(*list(full_model.backbone.children())[:-1])  # Remove fc layer
            # The classifier is the fc layer without sigmoid
            self.classifier = nn.Sequential(*list(full_model.backbone.fc.children())[:-1])  # Remove sigmoid
            # Add flatten for the backbone output [batch, 2048, 1, 1] -> [batch, 2048]
            self.flatten = nn.Flatten()

    def forward(self, x):
        # x is [batch, 3, 224, 224] - single frame
        if self.has_lstm:
            # ResNeXt-LSTM path
            features = self.backbone(x)  # [batch, 2048]
            # Add sequence dimension for LSTM
            features = features.unsqueeze(1)  # [batch, 1, 2048]
            lstm_out, _ = self.lstm(features)  # [batch, 1, hidden*2]
            final_feat = lstm_out[:, -1, :]  # [batch, hidden*2]
        else:
            # ImageClassifier path
            features = self.backbone(x)  # [batch, 2048, 1, 1] - features before fc
            final_feat = self.flatten(features)  # [batch, 2048]
        logits = self.classifier(final_feat)  # [batch, 1] - logits
        return logits

class GradCAMExplainer:
    def __init__(self, model):
        """
        Initialize GradCAM explainer for the ResNeXt backbone
        """
        self.model = model
        # Use the last convolutional layer of ResNeXt
        self.target_layers = [model.backbone.layer4[-1]]

        # Create a single-frame model that shares the backbone
        self.single_frame_model = SingleFrameModel(model)

        # Initialize GradCAM with the single frame model
        self.cam = GradCAM(
            model=self.single_frame_model,
            target_layers=self.target_layers
        )

    def generate_heatmap(self, frame_tensor, target_class=1):
        """
        Generate GradCAM heatmap for a single frame

        Args:
            frame_tensor: torch.Tensor [1, 3, 224, 224]
            target_class: int, 1 for fake, 0 for real

        Returns:
            heatmap: np.array [224, 224]
        """
        self.single_frame_model.eval()
        self.single_frame_model.zero_grad()

        # Keep as tensor for GradCAM
        input_tensor = frame_tensor
        input_tensor.requires_grad_(True)

        # Ensure input is on the same device as the model
        device = next(self.single_frame_model.parameters()).device
        input_tensor = input_tensor.to(device)

        # For binary classification, target_class=0 for real (low prob), 1 for fake (high prob)
        # But since output is logits, use ClassifierOutputTarget with the class index
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

        # Get the heatmap
        heatmap = grayscale_cam[0]  # Shape: [224, 224]

        return heatmap

    def generate_guided_backprop(self, frame_tensor, target_class=1):
        """
        Generate Guided Backpropagation visualization

        Args:
            frame_tensor: torch.Tensor [1, 3, 224, 224]
            target_class: int, 1 for fake, 0 for real

        Returns:
            guided_grad: np.array [224, 224, 3]
        """
        self.single_frame_model.eval()

        input_tensor = frame_tensor.clone().detach().requires_grad_(True)
        device = next(self.single_frame_model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Use GuidedBackpropReLUModel
        guided_model = GuidedBackpropReLUModel(self.single_frame_model, device)
        guided_grad = guided_model(input_tensor, target_category=target_class)

        # Normalize to [0,1]
        guided_grad = (guided_grad - guided_grad.min()) / (guided_grad.max() - guided_grad.min() + 1e-8)
        return guided_grad[0].transpose(1, 2, 0)  # [H, W, C]

    def generate_shap_explanation(self, frame_tensor, background_frames=None):
        """
        Generate SHAP explanation for the prediction

        Args:
            frame_tensor: torch.Tensor [1, 3, 224, 224]
            background_frames: list of background tensors for SHAP

        Returns:
            shap_values: SHAP values
        """
        if background_frames is None:
            # Use random background
            background_frames = [torch.randn_like(frame_tensor) for _ in range(10)]

        def model_predict(x):
            with torch.no_grad():
                return self.single_frame_model(x).cpu().numpy()

        explainer = shap.GradientExplainer(model_predict, torch.stack(background_frames))
        shap_values = explainer.shap_values(frame_tensor.cpu())

        return shap_values[0][0]  # For binary, take first class

    def overlay_heatmap(self, frame, heatmap, alpha=0.5):
        """
        Overlay heatmap on original frame

        Args:
            frame: np.array [H, W, 3] or PIL Image or torch.Tensor
            heatmap: np.array [224, 224]
            alpha: float, transparency of heatmap

        Returns:
            overlay: np.array [H, W, 3]
        """

        if isinstance(frame, torch.Tensor):
            # Detach and convert to numpy
            frame = frame.detach().cpu().numpy()
            if frame.shape[0] == 3:  # [C, H, W]
                frame = frame.transpose(1, 2, 0)  # [H, W, C]
            frame = np.uint8(frame * 255)

        if hasattr(frame, 'convert'):
            frame = np.array(frame)

        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = np.float32(heatmap_colored) / 255

        frame_float = np.float32(frame) / 255

        overlay = heatmap_colored * alpha + frame_float * (1 - alpha)

        overlay = np.uint8(255 * overlay)

        return overlay

    def explain_prediction(self, frames, prediction_score):
        """
        Generate explanations for a video prediction

        Args:
            frames: list of frames (PIL Images or np arrays)
            prediction_score: float, model prediction (0-1)

        Returns:
            explanations: dict with heatmaps and suspicious regions
        """
        explanations = {
            'prediction_score': prediction_score,
            'is_fake': prediction_score > 0.5,
            'confidence': max(prediction_score, 1-prediction_score),
            'heatmaps': [],
            'suspicious_frames': []
        }

        for i, frame in enumerate(frames):
            if isinstance(frame, torch.Tensor):
                if frame.dim() == 3:
                    if frame.shape[0] == 3:  # [C, H, W]
                        frame_tensor = frame.unsqueeze(0)
                    else:  # [H, W, C]
                        frame_tensor = frame.permute(2, 0, 1).unsqueeze(0).float() / 255
                else:  # [1, C, H, W]
                    frame_tensor = frame
            elif isinstance(frame, np.ndarray):
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255
            else:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                frame_tensor = transform(frame).unsqueeze(0)

            # Generate heatmap
            heatmap = self.generate_heatmap(frame_tensor, target_class=1 if prediction_score > 0.5 else 0)

            if explanations['is_fake'] and np.mean(heatmap) > 0.2:  # Lower threshold for fake predictions
                explanations['suspicious_frames'].append({
                    'frame_index': i,
                    'heatmap': heatmap,
                    'activation_score': np.mean(heatmap)
                })

            explanations['heatmaps'].append(heatmap)

        return explanations

# Utility function for video processing
def extract_frames_from_video(video_path, max_frames=30, frame_rate=1):
    """
    Extract frames from video for analysis

    Args:
        video_path: str, path to video file
        max_frames: int, maximum frames to extract
        frame_rate: int, extract every Nth frame

    Returns:
        frames: list of np.array frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_count += 1

    cap.release()
    return frames


class AudioDeepfakeDetector:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        """
        Initialize audio deepfake detector using Wav2Vec2
        """
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary: real/fake
        self.model.eval()

    def preprocess_audio(self, audio_path, sample_rate=16000):
        """
        Preprocess audio file for model input

        Args:
            audio_path: str, path to audio file
            sample_rate: int, target sample rate

        Returns:
            input_values: torch.Tensor
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate)

        # Process with Wav2Vec2 processor
        input_values = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_values
        return input_values

    def predict_audio(self, audio_path):
        """
        Predict if audio is real or fake

        Args:
            audio_path: str, path to audio file

        Returns:
            classification: str, "Real" or "Fake"
            confidence: float, confidence score
        """
        input_values = self.preprocess_audio(audio_path)

        with torch.no_grad():
            logits = self.model(input_values).logits
            probs = torch.softmax(logits, dim=-1)
            fake_prob = probs[0][1].item()  # Assuming class 1 is fake

        if fake_prob > 0.5:
            classification = "Fake"
            confidence = fake_prob
        else:
            classification = "Real"
            confidence = 1 - fake_prob

        return classification, confidence

# Watermarking utilities
def embed_watermark(image, watermark_text="AEGIS_VERIFIED", alpha=0.1):
    """
    Embed invisible watermark in image

    Args:
        image: np.array [H, W, 3]
        watermark_text: str
        alpha: float, watermark strength

    Returns:
        watermarked_image: np.array [H, W, 3]
    """
    # Simple LSB watermarking (for demo)
    watermarked = image.copy()
    # Convert text to binary
    binary_text = ''.join(format(ord(c), '08b') for c in watermark_text)
    binary_text += '00000000'  # Null terminator

    idx = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):  # RGB channels
                if idx < len(binary_text):
                    # Modify LSB
                    pixel = image[i, j, k]
                    watermarked[i, j, k] = (pixel & ~1) | int(binary_text[idx])
                    idx += 1
                else:
                    break
            if idx >= len(binary_text):
                break
        if idx >= len(binary_text):
            break

    return watermarked

def verify_watermark(image, watermark_text="AEGIS_VERIFIED"):
    """
    Verify watermark in image

    Args:
        image: np.array [H, W, 3]
        watermark_text: str

    Returns:
        is_verified: bool
    """
    # Extract LSBs
    extracted_bits = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                extracted_bits.append(str(image[i, j, k] & 1))
                if len(extracted_bits) >= len(watermark_text) * 8 + 8:  # +8 for null
                    break
            if len(extracted_bits) >= len(watermark_text) * 8 + 8:
                break
        if len(extracted_bits) >= len(watermark_text) * 8 + 8:
            break

    # Convert to text
    extracted_text = ''
    for i in range(0, len(extracted_bits), 8):
        byte = ''.join(extracted_bits[i:i+8])
        if len(byte) == 8:
            char = chr(int(byte, 2))
            if char == '\0':
                break
            extracted_text += char

    return extracted_text == watermark_text

if __name__ == "__main__":

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from models.resnext_lstm import DeepfakeDetector

    model = DeepfakeDetector()
    explainer = GradCAMExplainer(model)


    dummy_frame = torch.randn(1, 3, 224, 224)
    heatmap = explainer.generate_heatmap(dummy_frame)

    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: {heatmap.min():.3f} - {heatmap.max():.3f}")
