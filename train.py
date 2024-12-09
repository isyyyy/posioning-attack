import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist
import numpy as np
import os

# Tạo thư mục lưu mô hình nếu chưa có
os.makedirs("models", exist_ok=True)

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 2. Hàm đầu độc dữ liệu
def poison_data(x, y, target_label=7, source_label=1, poison_rate=0.8):
    """
    Đầu độc dữ liệu bằng cách thay đổi nhãn `source_label` thành `target_label`
    với một tỷ lệ nhất định (poison_rate).
    """
    poisoned_x, poisoned_y = x.copy(), y.copy()
    indices = np.where(y == source_label)[0]
    num_poisoned = int(len(indices) * poison_rate)
    poisoned_indices = np.random.choice(indices, num_poisoned, replace=False)
    poisoned_y[poisoned_indices] = target_label
    return poisoned_x, poisoned_y

# Dữ liệu bị đầu độc (tăng cường đầu độc)
poisoned_x_train, poisoned_y_train = poison_data(x_train, y_train, target_label=7, source_label=1, poison_rate=0.8)

# 3. Hàm xây dựng mô hình CNN
def create_model():
    """
    Xây dựng một mô hình CNN đơn giản cho bài toán phân loại ảnh.
    """
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # 10 lớp cho MNIST
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. Huấn luyện mô hình gốc
print("Training Original Model...")
original_model = create_model()
original_model.fit(x_train, y_train, epochs=5, batch_size=128)
original_model.save("models/original_model.h5")
print("Original Model Saved.")

# 5. Huấn luyện mô hình bị đầu độc
print("Training Poisoned Model...")
poisoned_model = create_model()
poisoned_model.fit(poisoned_x_train, poisoned_y_train, epochs=5, batch_size=128)
poisoned_model.save("models/poisoned_model.h5")
print("Poisoned Model Saved.")