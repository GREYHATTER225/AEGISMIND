from datasets.deepfake_dataset import DeepfakeDataset

ds = DeepfakeDataset('datasets/train')
print('Train samples:', len(ds))
labels = [s['label'] for s in ds.samples]
print('Real:', sum(labels), 'Fake:', len(labels) - sum(labels))

# Check val
ds_val = DeepfakeDataset('datasets/val')
print('Val samples:', len(ds_val))
labels_val = [s['label'] for s in ds_val.samples]
print('Val Real:', sum(labels_val), 'Val Fake:', len(labels_val) - sum(labels_val))
