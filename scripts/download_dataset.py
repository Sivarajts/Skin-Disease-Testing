import os
import requests
from pathlib import Path
import time
import zipfile
import shutil

def download_file(url, output_path):
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB blocks
        downloaded = 0
        start_time = time.time()
        
        print(f"Total file size: {total_size / (1024*1024*1024):.2f} GB")
        
        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                file.write(data)
                
                # Calculate progress
                progress = (downloaded / total_size) * 100
                elapsed_time = time.time() - start_time
                speed = downloaded / (1024*1024*elapsed_time)  # MB/s
                
                # Print progress
                print(f"\rDownloaded: {downloaded/(1024*1024*1024):.2f} GB / {total_size/(1024*1024*1024):.2f} GB ({progress:.1f}%) - Speed: {speed:.2f} MB/s", end='')
        
        print("\nDownload completed!")
        return True
    except Exception as e:
        print(f"\nError downloading from {url}: {e}")
        return False

def download_and_setup_dataset():
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/isic2019', exist_ok=True)
    
    print("Downloading ISIC 2019 dataset...")
    
    # Download ISIC 2019 dataset images
    isic_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
    isic_output = "data/isic2019.zip"
    
    # Download metadata
    metadata_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
    metadata_output = "data/isic2019/ISIC_2019_Training_GroundTruth.csv"
    
    if not os.path.exists(metadata_output):
        print("\nDownloading metadata...")
        if download_file(metadata_url, metadata_output):
            print("Metadata downloaded successfully!")
        else:
            print("Failed to download metadata")
            return False
    
    if not os.path.exists(isic_output):
        if download_file(isic_url, isic_output):
            print("ISIC 2019 dataset downloaded successfully!")
            
            # Extract the dataset
            print("Extracting ISIC 2019 dataset...")
            with zipfile.ZipFile(isic_output, 'r') as zip_ref:
                zip_ref.extractall('data/isic2019')
            
            # Clean up zip file
            os.remove(isic_output)
            print("Dataset extraction completed!")
        else:
            print("Failed to download ISIC 2019 dataset")
            return False
    else:
        print("ISIC 2019 dataset already exists!")
    
    print("\nDataset setup instructions:")
    print("1. The ISIC 2019 dataset has been downloaded and extracted to the 'data/isic2019' directory")
    print("2. Run the following command to organize the dataset:")
    print("   python scripts/organize_dataset.py")
    
    return True

if __name__ == "__main__":
    download_and_setup_dataset() 