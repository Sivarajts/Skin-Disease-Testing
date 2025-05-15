import os
import shutil
import random
from pathlib import Path
import pandas as pd

def organize_dataset():
    # Define disease classes
    disease_classes = [
        "acne",
        "allergy",
        "black_spots",
        "clear_skin",
        "dermatitis",
        "eczema",
        "melanoma",
        "not_skin_image",
        "psoriasis",
        "rosacea",
        "vitiligo"
    ]
    
    # Create necessary directories
    for split in ['train', 'val', 'test']:
        for disease in disease_classes:
            os.makedirs(f'data/{split}/{disease}', exist_ok=True)
    
    # Process ISIC 2019 dataset
    isic_dir = "data/isic2019/ISIC_2019_Training_Input"
    if os.path.exists(isic_dir):
        print("Organizing ISIC 2019 dataset...")
        
        # Read metadata
        metadata_path = os.path.join("data/isic2019", 'ISIC_2019_Training_GroundTruth.csv')
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            
            # Map ISIC classes to our disease classes
            class_mapping = {
                'MEL': 'melanoma',
                'NV': 'clear_skin',
                'BCC': 'dermatitis',
                'AKIEC': 'dermatitis',
                'BKL': 'dermatitis',
                'DF': 'dermatitis',
                'VASC': 'dermatitis'
            }
            
            # Organize images
            for _, row in df.iterrows():
                image_id = row['image']
                # Find the class with highest probability
                class_probs = row.drop('image').to_dict()
                max_class = max(class_probs.items(), key=lambda x: x[1])[0]
                target_class = class_mapping.get(max_class, 'not_skin_image')
                
                # Randomly assign to train/val/test
                rand = random.random()
                if rand < 0.7:
                    split = 'train'
                elif rand < 0.85:
                    split = 'val'
                else:
                    split = 'test'
                
                # Copy image to appropriate directory
                src = os.path.join(isic_dir, f'{image_id}.jpg')
                dst = f'data/{split}/{target_class}/{image_id}.jpg'
                if os.path.exists(src):
                    shutil.copy2(src, dst)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.capitalize()} set:")
        for disease in disease_classes:
            n_files = len(os.listdir(f'data/{split}/{disease}'))
            print(f"{disease}: {n_files} images")
    
    print("\nDataset organization completed successfully!")

if __name__ == "__main__":
    organize_dataset() 