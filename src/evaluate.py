import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Path to test dataset
TEST_PATH = "../data/test"

IMG_SIZE = 128
BATCH_SIZE = 32

# Load trained model
model = tf.keras.models.load_model("../models/kidney_model.h5")

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f"\nTest Accuracy: {accuracy}")
print(f"Test Loss: {loss}")

# Predictions
predictions = model.predict(test_data)
y_pred = (predictions > 0.5).astype(int).ravel()
y_true = test_data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_true, y_pred, target_names=test_data.class_indices.keys())
print("\nClassification Report:")
print(report)