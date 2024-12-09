import tensorflow as tf
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. Load MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# 2. Load mô hình
original_model = load_model("models/original_model.h5")
poisoned_model = load_model("models/poisoned_model.h5")


# 3. Hàm hiển thị confusion matrix
def plot_confusion_matrix(model, x_test, y_test, name):
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    conf_matrix = confusion_matrix(y_test, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{name} Confusion Matrix')
    plt.show()


print("Visualizing Original Model...")
plot_confusion_matrix(original_model, x_test, y_test, "Original Model")

print("\nVisualizing Poisoned Model...")
plot_confusion_matrix(poisoned_model, x_test, y_test, "Poisoned Model")


# 4. Hiển thị hình ảnh phân loại sai
def display_misclassified_images(model, x_test, y_test, name, num_images=10):
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    misclassified_indices = np.where(y_test != predicted_labels)[0]
    print(f"Number of misclassified images in {name}: {len(misclassified_indices)}")

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(misclassified_indices[:num_images]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_test[idx]}, Pred: {predicted_labels[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


print("Displaying Misclassified Images for Original Model...")
display_misclassified_images(original_model, x_test, y_test, "Original Model")

print("\nDisplaying Misclassified Images for Poisoned Model...")
display_misclassified_images(poisoned_model, x_test, y_test, "Poisoned Model")