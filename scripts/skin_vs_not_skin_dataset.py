import os
import requests
from PIL import Image
from io import BytesIO

# URLs for a few skin and not-skin images (public domain)
SKIN_IMAGE_URLS = [
    'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg',
    'https://images.pexels.com/photos/1130626/pexels-photo-1130626.jpeg',
    'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg',
    'https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg',
]

NOT_SKIN_IMAGE_URLS = [
    'https://images.pexels.com/photos/145939/pexels-photo-145939.jpeg',  # Tiger
    'https://images.pexels.com/photos/34950/pexels-photo.jpg',         # Landscape
    'https://images.pexels.com/photos/459225/pexels-photo-459225.jpeg',# Cat
    'https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg', # Car
]

def download_images(urls, folder):
    os.makedirs(folder, exist_ok=True)
    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img.save(os.path.join(folder, f'{i+1}.jpg'))
            print(f'Downloaded: {url}')
        except Exception as e:
            print(f'Failed to download {url}: {e}')

def main():
    download_images(SKIN_IMAGE_URLS, 'skin_vs_not_skin/skin')
    download_images(NOT_SKIN_IMAGE_URLS, 'skin_vs_not_skin/not_skin')
    print('Download complete!')

if __name__ == '__main__':
    main() 