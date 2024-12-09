import tensorflow as tf
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import os

# Tạo thư mục lưu kết quả nếu chưa có
os.makedirs("results", exist_ok=True)

# 1. Load MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# 2. Load mô hình
original_model = load_model("models/original_model.h5")
poisoned_model = load_model("models/poisoned_model.h5")


# 3. Đánh giá mô hình
def evaluate_model(model, x_test, y_test, name):
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == y_test) * 100
    report = classification_report(y_test, predicted_labels, output_dict=True)

    # Lưu báo cáo vào file CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"results/{name}_classification_report.csv")
    print(f"{name} Classification Report saved to results/{name}_classification_report.csv")

    return accuracy, report


# Đánh giá mô hình gốc
print("Evaluating Original Model...")
original_accuracy, original_report = evaluate_model(original_model, x_test, y_test, "Original_Model")

# Đánh giá mô hình bị đầu độc
print("\nEvaluating Poisoned Model...")
poisoned_accuracy, poisoned_report = evaluate_model(poisoned_model, x_test, y_test, "Poisoned_Model")

# 4. Chênh lệch độ chính xác
accuracy_diff = original_accuracy - poisoned_accuracy
print(f"\nAccuracy Difference: {accuracy_diff:.2f}%")

# 5. Lưu kết quả tổng hợp
summary = {
    "Model": ["Original", "Poisoned"],
    "Accuracy (%)": [original_accuracy, poisoned_accuracy],
    "Difference (%)": [0, accuracy_diff]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv("results/summary.csv", index=False)
print("Summary results saved to results/summary.csv")