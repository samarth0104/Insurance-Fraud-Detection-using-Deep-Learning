import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess the data
train_data = pd.read_csv("train.csv")
train_images_folder = "images/"  # Path to the images folder
train_labels = train_data["label"].astype(str)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_data["filename"], train_labels, test_size=0.2, random_state=42
)

# Create image data generators with increased data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,  # Increase rotation range
    width_shift_range=0.3,  # Adjust width shift range
    height_shift_range=0.3,  # Adjust height shift range
)

batch_size = 64

train_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame({"filename": train_images, "label": train_labels}),
    directory=train_images_folder,
    x_col="filename",
    y_col="label",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="binary",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

val_generator = val_datagen.flow_from_dataframe(
    pd.DataFrame({"filename": val_images, "label": val_labels}),
    directory=train_images_folder,
    x_col="filename",
    y_col="label",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="binary",
)

# Create ResNet50 model (pre-trained on ImageNet)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Unfreeze fewer layers for fine-tuning
for layer in base_model.layers[:-10]:
    layer.trainable = True

# Create a custom model with Batch Normalization and regularization
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))  # Adjust dropout rate
model.add(
    layers.Dense(
        512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )
)
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))  # Adjust dropout rate
model.add(layers.Dense(1, activation="sigmoid"))

# Compile the model with learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00001,  # Adjust initial learning rate
    decay_steps=10000,
    decay_rate=0.9,
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Implement early stopping
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Train the model with early stopping
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stopping],
)

# Evaluate on the test set

# Save the model for future use
model.save("insurance_claim_model_resnet50_opt.h5")
