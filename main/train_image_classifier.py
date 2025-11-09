import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.image_classifier import ImageClassifier
from datasets.image_dataset import BalancedImageDataset

def train_image_classifier(num_epochs=20, batch_size=32, learning_rate=1e-4,
                          save_path='models/pretrained/image_classifier.pt',
                          checkpoint_freq=5, patience=7):
    """
    Train the image classifier with balanced dataset and augmentation.

    Args:
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Initial learning rate
        save_path (str): Path to save the best model
        checkpoint_freq (int): Save checkpoint every N epochs
        patience (int): Early stopping patience
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # Create datasets
    train_dataset = BalancedImageDataset('datasets/train', augment=True)
    val_dataset = BalancedImageDataset('datasets/val', augment=False)

    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Class distribution - Real: {train_dataset.class_counts['real']}, Fake: {train_dataset.class_counts['fake']}")

    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return

    # Create data loaders with balanced sampling
    train_sampler = train_dataset.get_sampler()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = ImageClassifier().to(device)

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training tracking
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\n🎯 Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []

        train_progress = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_progress:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Skip invalid samples
            if torch.any(labels == -1):
                continue

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid before thresholding
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            train_preds.extend(predictions.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        train_precision = precision_score(train_labels, train_preds, zero_division=0)
        train_recall = recall_score(train_labels, train_preds, zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        val_progress = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                if torch.any(labels == -1):
                    continue

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid before thresholding
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                val_progress.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate validation metrics
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        # Confusion matrix
        if val_labels:
            tn, fp, fn, tp = confusion_matrix(val_labels, val_preds).ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        print(f"📈 Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"📈 Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"📊 Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to {save_path} (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = save_path.replace('.pt', '_checkpoint.pt')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved to {checkpoint_path}")

        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️ Early stopping triggered after {patience} epochs without improvement")
            break

    print("🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print(f"📁 Model saved at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Image Classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/pretrained/image_classifier.pt', help='Save path')

    args = parser.parse_args()

    train_image_classifier(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )
