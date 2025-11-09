import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ImageClassifier(nn.Module):
    """
    ResNet50-based image classifier for fake/real detection.
    Uses pre-trained weights and fine-tunes for binary classification.
    """
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(ImageClassifier, self).__init__()

        #  pre-trained ResNet50
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)


        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        x: Tensor [batch_size, 3, 224, 224]
        Returns: Tensor [batch_size] with probabilities
        """
        return self.backbone(x).squeeze()

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"Loaded weights from {path}")

# Test the model
if __name__ == "__main__":
    model = ImageClassifier()
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output: {output}")
