import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Original shape:")
print("Training:", x_train.shape)
print("Testing :", x_test.shape)

print("\nOriginal pixel range:")
print("Minimum:", x_train.min())
print("Maximum:", x_train.max())

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("\nAfter preprocessing:")
print("Training shape:", x_train.shape)
print("Testing shape :", x_test.shape)

print("\nAfter normalization:")
print("Minimum:", x_train.min())
print("Maximum:", x_train.max())

print("\nPreprocessing completed successfully!")