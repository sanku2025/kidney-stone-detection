import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Path to test dataset
TEST_PATH = "../data/test"

# Image size (must match training)
IMG_SIZE = 128
BATCH_SIZE = 32

# Load trained model
model = load_model("../models/kidney_model.h5")


# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)

print("\nTest Accuracy:", accuracy)
print("Test Loss:", loss)
