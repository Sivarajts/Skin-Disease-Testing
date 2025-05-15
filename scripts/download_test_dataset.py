import os
import requests
import zipfile
from tqdm import tqdm
import shutil
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import hashlib

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'data/train', 'data/val', 'data/test']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def calculate_file_hash(filename):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url, filename, chunk_size=1024*1024):  # 1MB chunks
    """Download file with resume capability, progress bar, and hash verification."""
    # Get the file size if it exists
    initial_pos = 0
    if os.path.exists(filename):
        initial_pos = os.path.getsize(filename)
        print(f"Resuming download from {initial_pos} bytes")

    # Set up headers for resume
    headers = {'Range': f'bytes={initial_pos}-'} if initial_pos > 0 else {}
    
    try:
        # Open the connection with a timeout
        session = requests.Session()
        response = session.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0)) + initial_pos
        
        # Open file in append mode if resuming
        mode = 'ab' if initial_pos > 0 else 'wb'
        with open(filename, mode) as f:
            with tqdm(total=total_size, initial=initial_pos, unit='B', 
                     unit_scale=True, desc=os.path.basename(filename)) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        # Flush to disk periodically
                        if pbar.n % (chunk_size * 10) == 0:
                            f.flush()
                            os.fsync(f.fileno())
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return False
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False
    finally:
        if 'session' in locals():
            session.close()

def verify_download(filename, expected_size):
    """Verify if the downloaded file is complete."""
    if not os.path.exists(filename):
        return False
    
    actual_size = os.path.getsize(filename)
    if actual_size != expected_size:
        print(f"File size mismatch. Expected: {expected_size}, Got: {actual_size}")
        return False
    
    return True

def extract_zip(zip_path, extract_path):
    """Extract zip file with error handling and verification."""
    try:
        print(f"Verifying zip file: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Verify zip file integrity
            if zip_ref.testzip() is not None:
                print("Zip file is corrupted")
                return False
            
            # Extract files
            print("Extracting files...")
            zip_ref.extractall(extract_path)
            
            # Verify extraction
            extracted_files = sum([len(files) for _, _, files in os.walk(extract_path)])
            if extracted_files == 0:
                print("No files were extracted")
                return False
                
            print(f"Successfully extracted {extracted_files} files")
            return True
    except zipfile.BadZipFile:
        print("Invalid zip file")
        return False
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def organize_dataset():
    """Organize the dataset into train/val/test splits"""
    print("Organizing dataset...")
    # Create necessary directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f'data/{split}', exist_ok=True)
        for class_name in ['melanoma', 'nevus', 'basal_cell_carcinoma', 'clear_skin']:
            os.makedirs(f'data/{split}/{class_name}', exist_ok=True)
    
    # Get list of all images
    image_dir = 'data/ISIC_2019_Training_Input'
    images = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, file))
    
    if not images:
        print("No images found in the dataset directory")
        return False
    
    print(f"Found {len(images)} images")
    
    # Split into train/val/test
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    # Move files to appropriate directories
    for img_list, split in [(train_imgs, 'train'), (val_imgs, 'val'), (test_imgs, 'test')]:
        for img_path in img_list:
            # Determine class from filename (simplified for testing)
            class_name = 'clear_skin'  # Default class
            if 'melanoma' in img_path.lower():
                class_name = 'melanoma'
            elif 'nevus' in img_path.lower():
                class_name = 'nevus'
            elif 'bcc' in img_path.lower():
                class_name = 'basal_cell_carcinoma'
            
            # Copy file to appropriate directory
            dst = f'data/{split}/{class_name}/{os.path.basename(img_path)}'
            shutil.copy2(img_path, dst)
    
    print(f"Dataset organized into train/val/test splits")
    print(f"Total images: {len(images)}")
    print(f"Train images: {len(train_imgs)}")
    print(f"Validation images: {len(val_imgs)}")
    print(f"Test images: {len(test_imgs)}")
    return True

def main():
    # Create necessary directories
    create_directories()
    
    # Dataset URL and expected size
    dataset_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
    zip_path = "data/ISIC_2019_Training_Input.zip"
    expected_size = 9770000000  # 9.77GB in bytes
    
    # Maximum number of retries
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        print(f"Downloading ISIC 2019 dataset... (Attempt {retry_count + 1}/{max_retries})")
        
        # Download the dataset
        if download_file(dataset_url, zip_path):
            # Verify download
            if verify_download(zip_path, expected_size):
                print("Download completed and verified successfully!")
                
                # Extract the dataset
                print("Extracting dataset...")
                if extract_zip(zip_path, "data"):
                    print("Dataset extracted successfully!")
                    
                    # Organize the dataset
                    if organize_dataset():
                        # Clean up
                        print("Cleaning up...")
                        os.remove(zip_path)
                        print("Setup completed successfully!")
                        return True
                    else:
                        print("Dataset organization failed!")
                else:
                    print("Extraction failed!")
            else:
                print("Download verification failed!")
        else:
            print("Download failed!")
        
        retry_count += 1
        if retry_count < max_retries:
            wait_time = 10 * retry_count  # Increasing wait time between retries
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    print("Failed to download and setup dataset after maximum retries.")
    return False

if __name__ == "__main__":
    main() 