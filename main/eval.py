import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from datasets.image_dataset import BalancedImageDataset
from models.image_classifier import ImageClassifier
from torchvision import transforms

def evaluate_model(data_dir, model_path, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transforms (same as training, but no augmentations)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    val_dataset = BalancedImageDataset(data_dir, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Evaluating on {len(val_dataset)} samples")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Skip invalid samples
            if torch.any(labels == -1):
                continue

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()  # Apply sigmoid to get probabilities
            preds = (probs > 0.5).astype(int)  # Threshold at 0.5

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print("\nEvaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': (tn, fp, fn, tp)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/classifier/val')
    parser.add_argument('--model_path', type=str, default='weights/face_classifier.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    evaluate_model(args.data_dir, args.model_path, args.batch_size)