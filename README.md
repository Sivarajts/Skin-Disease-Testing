# Skin Disease Detection System

A deep learning-based system for detecting various skin conditions using computer vision and machine learning.

## Features

- Upload and analyze skin images
- Detect multiple skin conditions:
  - Melanoma
  - Nevus (Moles)
  - Basal Cell Carcinoma
  - Clear Skin
- Detailed disease information and recommendations
- User-friendly web interface

## Setup Instructions

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download and prepare the dataset:
```bash
python scripts/download_test_dataset.py
```

3. Train the model:
```bash
python scripts/train_model.py
```

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `scripts/`: Contains utility scripts
  - `download_test_dataset.py`: Downloads and prepares the dataset
  - `train_model.py`: Trains the deep learning model
  - `reset_db.py`: Resets the database
- `models/`: Stores trained models
- `data/`: Contains the dataset
  - `train/`: Training images
  - `val/`: Validation images
  - `test/`: Test images

## Model Architecture

The system uses EfficientNetB0 as the base model with custom classification layers for skin disease detection. The model is trained on the ISIC 2019 dataset and fine-tuned for our specific use case.

## Note

This system is for preliminary screening only. Always consult a healthcare professional for proper diagnosis and treatment.

## Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_email_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
FAST2SMS_API_KEY=your_fast2sms_api_key
```

**Do NOT commit your real `.env` file to GitHub!**
