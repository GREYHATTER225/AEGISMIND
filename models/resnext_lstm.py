import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetector(nn.Module):
    def __init__(self, lstm_hidden_size=512, lstm_layers=2, bidirectional=True):
        super(DeepfakeDetector, self).__init__()
        
        # ResNeXt backbone
        self.backbone = models.resnext50_32x4d(pretrained=True)
        self.backbone.fc = nn.Identity()  # remove classifier, keep feature vector
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Classifier
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 1)
            # Remove Sigmoid here since we apply it in inference
        )

    def forward(self, frames):
        """
        frames: Tensor [batch_size, num_frames, 3, 224, 224]
        """
        batch_size, num_frames, _, _, _ = frames.shape
        batch_features = []
        for b in range(batch_size):
            video_features = []
            for f in range(num_frames):
                frame = frames[b, f].unsqueeze(0)  # [1, 3, 224, 224]
                feat = self.backbone(frame)  # [1, 2048]
                video_features.append(feat.squeeze(0))  # [2048]
            video_features = torch.stack(video_features, dim=0)  # [num_frames, 2048]
            batch_features.append(video_features)
        batch_features = torch.stack(batch_features, dim=0)  # [batch_size, num_frames, 2048]

        lstm_out, _ = self.lstm(batch_features)  # [batch_size, num_frames, hidden*2]
        final_feat = lstm_out[:, -1, :]  # take last output [batch_size, hidden*2]
        out = self.classifier(final_feat)  # [batch_size, 1]
        return out.squeeze(1)  # [batch_size]

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"Loaded weights from {path}")

# Dummy test
if __name__ == "__main__":
    model = DeepfakeDetector()
    dummy_frames = torch.randn(1, 5, 3, 224, 224)  # batch_size=1, num_frames=5
    output = model(dummy_frames)
    print("Dummy output:", output)
