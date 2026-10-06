import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype("float32") / 255.0

x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.models.load_model("models/digit_cnn_model.keras")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\nModel Evaluation Results")
print("------------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

predictions = model.predict(x_test, verbose=1)

y_pred = predictions.argmax(axis=1)

print("\nClassification Report")
print("---------------------")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("MNIST Digit Recognition - Confusion Matrix")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")

plt.tight_layout()

plt.savefig("screenshots/confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved successfully!")