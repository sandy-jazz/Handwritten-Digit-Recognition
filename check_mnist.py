import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("\nMNIST Dataset Loaded Successfully!")

print("\nTraining data:")
print("Images:", x_train.shape)
print("Labels:", y_train.shape)

print("\nTesting data:")
print("Images:", x_test.shape)
print("Labels:", y_test.shape)

print("\nPixel value range:")
print("Minimum:", x_train.min())
print("Maximum:", x_train.max())

print("\nFirst image label:", y_train[0])