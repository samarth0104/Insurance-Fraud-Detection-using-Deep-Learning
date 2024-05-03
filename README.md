# Vehicle Damage Detection Using ResNet50

This project utilizes the ResNet50 deep learning architecture to detect fraudulent vehicle insurance claims. The dataset consists of a folder containing images and two CSV files for training and testing the model.

## Project Structure

/images/ - Folder containing all the training and testing images.
train.csv - CSV file with filenames and labels for training data.
test.csv - CSV file with filenames for testing the model.

## Dependencies

Python 3.x
TensorFlow 2.x
Keras
NumPy
Pandas
scikit-learn

## You can install the necessary libraries using pip:

pip install tensorflow keras numpy pandas scikit-learn

## Training the Model

To train the model, run the train.py script. This script will load the data from train.csv, preprocess the images, compile the ResNet50 model, and fit the model on the training data.

## How to run
python main.py

## Sample Code - main.py
python

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd

## Load and preprocess data
data = pd.read_csv('train.csv')
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_dataframe(dataframe=data, directory='images/',
                                              x_col="filename", y_col="label",
                                              class_mode="binary", subset="training")

## Build the model
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

## Compile model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

## Train model
model.fit(train_generator, epochs=10)
model.save('vehicle_damage_detector.h5')
Testing the Model

To evaluate the model's performance, use the test.py script. This script loads the trained model, processes images from test.csv, and evaluates them using the F1 score.


python test.py
Sample Code - test.py
python
Copy code
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.metrics import classification_report

## Load model
model = tf.keras.models.load_model('vehicle_damage_detector.h5')

## Load and preprocess test data
test_data = pd.read_csv('test.csv')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_data, directory='images/',
                                                  x_col="filename", y_col=None,
                                                  class_mode=None, shuffle=False)

## Predict
predictions = model.predict(test_generator)
predictions = [1 if x > 0.5 else 0 for x in predictions]

## Evaluate model
report = classification_report(test_data['label'], predictions, target_names=['Non-Fraudulent', 'Fraudulent'])
print(report)
