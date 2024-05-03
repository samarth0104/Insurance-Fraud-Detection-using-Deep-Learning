import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import random

# Load and preprocess the data
train_data = pd.read_csv("train.csv")
train_images_folder = "images"  # Path to the images folder
# Convert the values in the 'label' column to strings
train_data["label"] = train_data["label"].astype(str)

# Randomly select 1000 images from the training set for testing
test_images = random.sample(train_data["filename"].tolist(), 1000)

# Create an image data generator for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    validation_split=0.2,  # Splitting data into train and validation sets
)

# Generate training and validation data
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=train_images_folder,
    x_col="filename",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training",
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=train_images_folder,
    x_col="filename",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
)

# Create VGG16 model (pre-trained on ImageNet)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of VGG16
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator),
)

# Evaluate the model on the random 1000 training images
train_test_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_test_generator = train_test_datagen.flow_from_dataframe(
    pd.DataFrame({"filename": test_images}),
    directory=train_images_folder,
    x_col="filename",
    y_col=None,  # No labels needed for testing
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # No class mode needed
    shuffle=False,
)

model.save("main_vgg16.h5")
