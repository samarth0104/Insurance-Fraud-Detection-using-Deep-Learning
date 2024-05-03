import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load and preprocess the data
train_data = pd.read_csv("train.csv")
train_images_folder = "images/"
train_labels = train_data["label"].astype(str)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_data["filename"], train_labels, test_size=0.2, random_state=42
)

# Create image data generators with increased data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    preprocessing_function=preprocess_input  # EfficientNet-specific preprocessing
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input  # EfficientNet-specific preprocessing
)

# Create EfficientNetB0 base model (pre-trained on ImageNet)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create a custom model on top of the pre-trained EfficientNetB0
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model with learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.9
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Create data generators
batch_size = 32
train_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame({'filename': train_images, 'label': train_labels}),
    directory=train_images_folder,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_dataframe(
    pd.DataFrame({'filename': val_images, 'label': val_labels}),
    directory=train_images_folder,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

# Train the model with early stopping
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Save the model
model.save("new_insurance_claim_model_optimized.h5")
