import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

def load_data(folder_path):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        img = cv2.resize(img, (128, 128))  # Resize images to a fixed size
        images.append(img)
        labels.append(1 if "Y" in filename else 0)
    return np.array(images), np.array(labels)

yes_images, yes_labels = load_data("dataset/yes")
no_images, no_labels = load_data("dataset/no")

# Combine the datasets
images = np.concatenate((yes_images, no_images), axis=0)
labels = np.concatenate((yes_labels, no_labels), axis=0)

# Normalize pixel values to the range [0, 1]
images = images / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Loss: {loss}, Accuracy: {accuracy}")

model.save('brain_tumor_model.json')