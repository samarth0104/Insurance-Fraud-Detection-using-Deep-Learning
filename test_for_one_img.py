import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the trained model
model = load_model("main.h5")

# Load and preprocess the image
image_path = "C:\\Users\\samar\\OneDrive\\Desktop\\PESU\\Extra\\Fraud_detection_523_505_502_548\\images\\fraud.jpg"  # Replace with the path to your image
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = img_array / 255.0  # Rescale the pixel values to [0, 1]

# Expand the dimension to match the input shape of the model
img_array = np.expand_dims(img_array, axis=0)

# Predict the output
prediction = model.predict(img_array)


binary_prediction = (prediction > 0.075).astype(int)
print(binary_prediction)
