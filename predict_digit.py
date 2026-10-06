import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the trained CNN model
model = tf.keras.models.load_model("models/digit_cnn_model.keras")

print("Handwritten Digit Recognition")
print("-----------------------------")

# Ask the user for an image path
image_path = input("Enter the path of your handwritten digit image: ")

# Check whether the file exists
if not os.path.exists(image_path):
    print("Image file not found.")
    exit()

# Open the image
image = Image.open(image_path)

# Convert image to grayscale
image = image.convert("L")

# Resize image to MNIST size
image = image.resize((28, 28))

# Convert image to numpy array
image_array = np.array(image)

# Normalize pixel values
image_array = image_array.astype("float32") / 255.0

# Reshape image for CNN
image_array = image_array.reshape(1, 28, 28, 1)

# Make prediction
prediction = model.predict(image_array, verbose=0)

# Find the digit with the highest probability
predicted_digit = np.argmax(prediction[0])

# Get confidence score
confidence = np.max(prediction[0]) * 100

print("\nPrediction Result")
print("-----------------")
print("Predicted Digit :", predicted_digit)
print(f"Confidence      : {confidence:.2f}%")