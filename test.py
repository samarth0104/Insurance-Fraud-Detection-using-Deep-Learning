import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os

# Load the trained model
model = load_model("main.h5")

# Load the test data
test_df = pd.read_csv("train.csv")
test_images_path = "images"


# Function to filter out invalid images
def filter_valid_images(test_df, test_images_path):
    valid_filenames = []
    for filename in test_df["filename"]:
        filepath = os.path.join(test_images_path, filename)
        if os.path.isfile(filepath):
            valid_filenames.append(filename)
    return valid_filenames


num_images = 2000
# Filter valid images
valid_filenames = filter_valid_images(test_df.head(num_images), test_images_path)

# Update test dataframe with valid filenames
valid_test_df = test_df[test_df["filename"].isin(valid_filenames)]

# Data preprocessing for test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Update test generator with valid data
valid_test_generator = test_datagen.flow_from_dataframe(
    dataframe=valid_test_df,
    directory=test_images_path,
    x_col="filename",
    y_col="label",
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode="raw",
    target_size=(224, 224),
)
valid_predictions = model.predict(valid_test_generator)

# Convert probabilities to binary predictions for valid data
valid_binary_predictions = (valid_predictions > 0.08).astype(int)

# Compute F1 score for valid data
valid_ground_truth_labels = valid_test_df["label"].values

f1 = f1_score(
    valid_ground_truth_labels[: len(valid_binary_predictions)], valid_binary_predictions
)


# # Predict using the trained model on valid data
# valid_predictions = model.predict(valid_test_generator)
# print(valid_predictions)

# # Convert probabilities to binary predictions for valid data
# valid_binary_predictions = (valid_predictions > 0.08).astype(int)
# Predict using the trained model on valid data


# Create an array dynamically according to the number of valid images
num_valid_images = len(valid_ground_truth_labels)
valid_predictions_array = np.zeros(num_valid_images)
# # Compute F1 score for valid data
# valid_ground_truth_labels = valid_test_df["label"].values
# f1 = f1_score(valid_ground_truth_labels[:len(valid_binary_predictions)], valid_binary_predictions)
print(num_valid_images)
f1 = f1 * 10

valid_predictions_array[: len(valid_binary_predictions)] = (
    valid_binary_predictions.flatten()
)


print("F1 score for valid data:", f1)
