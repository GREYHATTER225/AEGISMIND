import os

real_dir = 'datasets/train/real'
fake_dir = 'datasets/train/fake'

real_files = os.listdir(real_dir)
fake_files = os.listdir(fake_dir)

print(f'Real files: {len(real_files)}')
print(f'Fake files: {len(fake_files)}')

real_images = [f for f in real_files if f.endswith(('.jpg', '.jpeg', '.png'))]
fake_images = [f for f in fake_files if f.endswith(('.jpg', '.jpeg', '.png'))]

print(f'Real images: {len(real_images)}')
print(f'Fake images: {len(fake_images)}')

# Check extracted images
real_images_dir = 'datasets/train/real_images'
fake_images_dir = 'datasets/train/fake_images'

if os.path.exists(real_images_dir):
    real_extracted = os.listdir(real_images_dir)
    print(f'Real extracted images: {len(real_extracted)}')

if os.path.exists(fake_images_dir):
    fake_extracted = os.listdir(fake_images_dir)
    print(f'Fake extracted images: {len(fake_extracted)}')

if real_files:
    print(f'Sample real file: {real_files[0]}')
if fake_files:
    print(f'Sample fake file: {fake_files[0]}')
