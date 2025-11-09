import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load real faces
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img_file in os.listdir(real_dir):
                if img_file.endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(real_dir, img_file), 0))  # 0 = real

        # Load AI faces
        ai_dir = os.path.join(root_dir, 'ai')
        if os.path.exists(ai_dir):
            for img_file in os.listdir(ai_dir):
                if img_file.endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(ai_dir, img_file), 1))  # 1 = ai

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_model(model_name='resnet50'):
    if model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    return model

def train_classifier(data_dir, model_name='resnet50', epochs=20, batch_size=32, lr=1e-4, save_path='weights/face_classifier.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = FaceDataset(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = FaceDataset(os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        model.train()
        train_preds, train_labels = [], []
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        # Validate
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Prob of AI
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs)

        # Metrics
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0

        print(".4f")
        print(".4f")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/classifier')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='weights/face_classifier.pth')
    args = parser.parse_args()

