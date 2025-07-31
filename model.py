import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import sys
import logging

# Fix the encoding issue by setting stdout to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Set up logging to a file to capture any potential errors
logging.basicConfig(filename='training_log.txt', level=logging.INFO)

# Paths to the train and test directories
train_dir = r'C:\Users\imroz\Music\wastecheck\TRAIN'  # Replace with your actual path
test_dir = r'C:\Users\imroz\Music\wastecheck\TEST'  # Replace with your actual path

# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Data Augmentation and Image Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split for validation data
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output (0 or 1)
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Log the model summary
logging.info("Model Summary:\n" + str(model.summary()))

# Train the model
try:
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=[early_stopping]
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_data)
    logging.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Make predictions and generate classification report
    predictions = model.predict(test_data)
    predicted_classes = (predictions > 0.5).astype("int32")
    true_classes = test_data.classes
    class_labels = list(test_data.class_indices.keys())

    # Classification report and confusion matrix
    class_report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    logging.info(f"Classification Report:\n{class_report}")
    print(class_report)

    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    logging.info(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

    # Save the trained model for future use
    model.save('waste_classifier_model.h5')
    logging.info("Model saved as 'waste_classifier_model.h5'")

except Exception as e:
    logging.error(f"Error occurred during training: {str(e)}")
    print(f"Error occurred: {str(e)}")
