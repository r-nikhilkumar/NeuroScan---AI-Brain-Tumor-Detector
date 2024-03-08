from tensorflow.keras.models import load_model
import numpy as np
import cv2


# Load the model
# model = load_model('brain_tumor_detector_model.h5')
model = load_model('brain_tumor_model.json')

# Load and preprocess the new image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Assuming your model expects input size of 128x128
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Make a prediction
image_path = "no 4.jpg"
preprocessed_image = preprocess_image(image_path)
prediction = model.predict(preprocessed_image)

# Assuming your model predicts binary classification (0 or 1)
if prediction[0][0] > 0.5:
    print("Tumor detected")
else:
    print("No tumor detected")
# print(prediction)