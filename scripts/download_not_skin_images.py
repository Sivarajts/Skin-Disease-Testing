import os
import requests
from PIL import Image
from io import BytesIO
import random

UNSPLASH_URL = 'https://source.unsplash.com/random/256x256'
NUM_IMAGES = 200

out_dirs = [
    ('data/train/not_skin_image', 0.7),
    ('data/val/not_skin_image', 0.15),
    ('data/test/not_skin_image', 0.15)
]

# Calculate split counts
splits = [int(NUM_IMAGES * frac) for _, frac in out_dirs]
splits[-1] = NUM_IMAGES - sum(splits[:-1])  # Ensure total is NUM_IMAGES

# Download images
all_images = []
for i in range(NUM_IMAGES):
    try:
        response = requests.get(UNSPLASH_URL, timeout=10)
        if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('image'):
            img = Image.open(BytesIO(response.content)).convert('RGB')
            all_images.append(img)
            print(f'Downloaded image {i+1}/{NUM_IMAGES}')
        else:
            print(f'Skipped non-image response for image {i+1}')
    except Exception as e:
        print(f'Failed to download image {i+1}: {e}')

# Shuffle and split
random.shuffle(all_images)
start = 0
for (out_dir, _), count in zip(out_dirs, splits):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(count):
        idx = start + i
        all_images[idx].save(os.path.join(out_dir, f'{idx+1}.jpg'))
    start += count

print('Downloaded and split random not_skin_image images!') 