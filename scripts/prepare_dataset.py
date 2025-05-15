import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import gdown
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Dataset URLs and configurations
DATASET_CONFIGS = {
    "HAM10000": {
        "type": "kaggle",
        "dataset": "kmader/skin-cancer-mnist-ham10000",
        "files": ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]
    },
    "ISIC2019": {
        "type": "kaggle",
        "dataset": "shayanfaghihi/isic2019",
        "files": ["ISIC_2019_Training_Input.zip"]
    },
    "DermNet": {
        "type": "gdrive",
        "url": "https://drive.google.com/uc?id=1-0C7yqX5qX5qX5qX5qX5qX5qX5qX5qX5",
        "filename": "dermnet.zip"
    }
}

def create_directories():
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    # Create class directories
    classes = [
        "Clear Skin",
        "Acne",
        "Scars",
        "Dark Spots",
        "Allergy",
        "Eczema",
        "Psoriasis",
        "Melanoma",
        "Basal Cell Carcinoma",
        "Squamous Cell Carcinoma"
    ]
    
    for split in ["train", "val", "test"]:
        for class_name in classes:
            os.makedirs(f"data/{split}/{class_name}", exist_ok=True)

def download_kaggle_dataset(dataset_name, config):
    """Download dataset from Kaggle"""
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset files
        for file in config['files']:
            api.dataset_download_file(
                config['dataset'],
                file,
                path='data'
            )
            print(f"Downloaded {file}")
            
    except Exception as e:
        print(f"Error downloading {dataset_name} dataset: {str(e)}")

def download_gdrive_file(url, filename):
    """Download file from Google Drive"""
    try:
        output = f"data/{filename}"
        gdown.download(url, output, quiet=False)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Error downloading file: {str(e)}")

def extract_zip(zip_path, extract_path):
    """Extract zip file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted {zip_path} to {extract_path}")
    except Exception as e:
        print(f"Error extracting {zip_path}: {str(e)}")

def organize_dataset():
    """Organize the downloaded datasets"""
    # Create a DataFrame to track images and their labels
    data = []
    
    # Process each dataset
    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"\nProcessing {dataset_name} dataset...")
        
        if config['type'] == 'kaggle':
            download_kaggle_dataset(dataset_name, config)
        elif config['type'] == 'gdrive':
            download_gdrive_file(config['url'], config['filename'])
        
        # Extract downloaded files
        for file in config['files']:
            zip_path = f"data/{file}"
            extract_path = f"data/{dataset_name}"
            extract_zip(zip_path, extract_path)
        
        # Organize images into class folders
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    # Get class from filename or directory structure
                    class_name = os.path.basename(os.path.dirname(os.path.join(root, file)))
                    if class_name not in DISEASE_CLASSES:
                        class_name = "Clear Skin"  # Default class
                    
                    data.append({
                        'image_path': os.path.join(root, file),
                        'class': class_name
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Move files to appropriate directories
    for df, split in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        for _, row in df.iterrows():
            src = row['image_path']
            dst = f"data/{split}/{row['class']}/{os.path.basename(src)}"
            shutil.copy2(src, dst)
    
    # Save dataset information
    df.to_csv('data/dataset_info.csv', index=False)
    print("\nDataset organization complete!")

if __name__ == "__main__":
    print("Starting dataset preparation...")
    create_directories()
    organize_dataset()
    print("Dataset preparation completed!") 