import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0  # Using B0 for faster training
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import time
from datetime import datetime, timedelta
import gc
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Optimized Configuration for 8-hour training
IMG_SIZE = (160, 160)  # Reduced size for faster training
BATCH_SIZE = 32  # Increased batch size for faster training
EPOCHS = 50  # Reduced epochs but with better optimization
LEARNING_RATE = 0.001  # Increased learning rate
DROPOUT_RATE = 0.3  # Reduced dropout for faster convergence

# Extended Disease classes
DISEASE_CLASSES = [
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

# Enable memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(f"GPU memory growth error: {e}")

def get_present_classes(train_dir):
    """Get list of classes that have images in the training directory."""
    try:
        classes = [d for d in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, d)) 
                  and len(os.listdir(os.path.join(train_dir, d))) > 0]
        classes.sort()
        return classes
    except Exception as e:
        logging.error(f"Error getting present classes: {str(e)}")
        raise

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_times = []
        self.best_val_acc = 0
        self.current_epoch = 0
        self.last_save_time = time.time()
        self.save_interval = 300  # Save progress every 5 minutes

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        logging.info("\n=== Training Started ===")
        logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Total epochs: {self.total_epochs}")
        logging.info(f"Batch size: {BATCH_SIZE}")
        logging.info(f"Image size: {IMG_SIZE}")
        expected_end = datetime.now() + timedelta(hours=8)
        logging.info(f"Expected completion time: {expected_end.strftime('%Y-%m-%d %H:%M:%S')}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        logging.info(f"\nEpoch {epoch + 1}/{self.total_epochs}")

    def on_batch_end(self, batch, logs=None):
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            try:
                self.model.save('models/checkpoints/auto_save.h5')
                self.last_save_time = current_time
                logging.info(f"Auto-saved checkpoint at batch {batch}")
            except Exception as e:
                logging.error(f"Error saving auto-checkpoint: {str(e)}")

    def on_epoch_end(self, epoch, logs=None):
        try:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            avg_epoch_time = np.mean(self.epoch_times)
            remaining_epochs = self.total_epochs - (epoch + 1)
            estimated_time_remaining = avg_epoch_time * remaining_epochs
            
            current_val_acc = logs.get('val_accuracy', 0)
            if current_val_acc > self.best_val_acc:
                self.best_val_acc = current_val_acc
                self.model.save('models/best_model.h5')
                logging.info(f"New best model saved with accuracy: {self.best_val_acc:.4f}")
            
            completion_time = datetime.now() + timedelta(seconds=int(estimated_time_remaining))
            
            logging.info(f"\nEpoch {epoch + 1} Summary:")
            logging.info(f"Time taken: {epoch_time:.1f} seconds")
            logging.info(f"Training accuracy: {logs.get('accuracy'):.4f}")
            logging.info(f"Validation accuracy: {current_val_acc:.4f}")
            logging.info(f"Best validation accuracy so far: {self.best_val_acc:.4f}")
            logging.info(f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}")
            logging.info(f"Estimated completion time: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Save progress
            self.model.save('models/checkpoints/last_checkpoint.h5')
            
            # Clear memory
            gc.collect()
            tf.keras.backend.clear_session()
            
        except Exception as e:
            logging.error(f"Error in epoch end callback: {str(e)}")
            # Save emergency checkpoint
            try:
                self.model.save('models/checkpoints/emergency_save.h5')
                logging.info("Emergency checkpoint saved")
            except:
                logging.error("Failed to save emergency checkpoint")

def create_model(num_classes):
    try:
        logging.info("\nCreating model...")
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        
        # Unfreeze last 15 layers for fine-tuning
        for layer in base_model.layers[-15:]:
            layer.trainable = True
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(DROPOUT_RATE)(x)
        
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        logging.error(f"Error creating model: {str(e)}")
        raise

def create_data_generators(present_classes):
    try:
        logging.info("\nSetting up data generators...")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        logging.info("Loading training data...")
        train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=present_classes,
            shuffle=True
        )

        logging.info("Loading validation data...")
        validation_generator = test_datagen.flow_from_directory(
            'data/val',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=present_classes,
            shuffle=False
        )

        logging.info("\nDataset Statistics:")
        logging.info(f"Training samples: {len(train_generator.filenames)}")
        logging.info(f"Validation samples: {len(validation_generator.filenames)}")
        
        class_counts = train_generator.classes
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(class_counts),
            y=class_counts
        )
        class_weights = dict(enumerate(class_weights))
        
        return train_generator, validation_generator, class_weights
    except Exception as e:
        logging.error(f"Error creating data generators: {str(e)}")
        raise

def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.title('F1 Score')
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()

def evaluate_model(model, test_generator):
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Generate detailed classification report
    report = classification_report(y_true, y_pred, target_names=DISEASE_CLASSES)
    print("\nDetailed Classification Report:")
    print(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DISEASE_CLASSES,
                yticklabels=DISEASE_CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()

def train_model():
    try:
        logging.info("\n=== Starting Model Training ===")
        start_time = time.time()
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/checkpoints', exist_ok=True)
        
        # Get present classes
        present_classes = get_present_classes('data/train')
        logging.info(f"\nTraining on classes: {present_classes}")
        
        # Check for existing checkpoint
        checkpoint_path = 'models/checkpoints/last_checkpoint.h5'
        initial_epoch = 0
        
        if os.path.exists(checkpoint_path):
            logging.info("\nFound existing checkpoint. Resuming training...")
            model = load_model(checkpoint_path)
            # Get the last epoch from the checkpoint filename
            try:
                checkpoint_files = [f for f in os.listdir('models/checkpoints') if f.startswith('epoch_')]
                if checkpoint_files:
                    last_epoch = max([int(f.split('_')[1].split('.')[0]) for f in checkpoint_files])
                    initial_epoch = last_epoch + 1
            except Exception as e:
                logging.warning(f"Could not determine last epoch, starting from 0: {str(e)}")
            logging.info(f"Resuming from epoch {initial_epoch}")
        else:
            model = create_model(len(present_classes))
        
        train_generator, validation_generator, class_weights = create_data_generators(present_classes)
        
        # Create checkpoint callback with epoch number in filename
        checkpoint_callback = ModelCheckpoint(
            filepath='models/checkpoints/epoch_{epoch:02d}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        callbacks = [
            checkpoint_callback,
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            ProgressCallback(EPOCHS)
        ]
        
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        model.save('models/final_model.h5')
        logging.info("Training completed successfully!")
        
        return history
        
    except Exception as e:
        logging.error(f"\nError during training: {str(e)}")
        # Try to save emergency checkpoint
        try:
            if 'model' in locals():
                model.save('models/checkpoints/emergency_save.h5')
                logging.info("Emergency checkpoint saved")
        except:
            logging.error("Failed to save emergency checkpoint")
        raise

if __name__ == "__main__":
    try:
        logging.info("Starting model training...")
        history = train_model()
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)
