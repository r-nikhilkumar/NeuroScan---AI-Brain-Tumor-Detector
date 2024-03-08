from django.shortcuts import render
from django.http import HttpResponse
from tensorflow.keras.models import load_model
from django.middleware.csrf import get_token
import numpy as np
import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from PIL import Image
import os

def getCSRF(request):
    token = get_token(request)
    # data = {'message': token}
    # return HttpResponse(token)
    return JsonResponse({'message': token})


# Load the model
model = load_model('C:\\Users\\theni\\Desktop\\AI\\backend\\brain_tumor_detector_model.h5')
# model = load_model('brain_tumor_detector_model.h5')

# Load and preprocess the new image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    print("img-> ",img)
    img = cv2.resize(img, (128, 128))  # Assuming your model expects input size of 128x128
    print("img-> ",img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

def save_image_to_folder(image_path, image):
    try:
        # Save the image to the specified path
        image.save(image_path)
        print("Image saved successfully to:", image_path)
    except Exception as e:
        print("Error saving image:", str(e))


@csrf_exempt
def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        Img = Image.open(image)
        save_path = 'C:/Users/theni/Desktop/AI/backend/predict/images/'+image.name
        save_image_to_folder(save_path, Img)
        myimg = 'C:\\Users\\theni\\Desktop\\AI\\backend\\predict\\images\\'+image.name
        procImg = preprocess_image(myimg)
        pred = model.predict(procImg)
        
        # Delete the image file from the filesystem
        if os.path.exists(myimg):
            os.remove(myimg)
        
        if pred[0][0] > 0.5:
                return JsonResponse({'message': "Tumor Detected"}, status=200)
        else:
            return JsonResponse({'message': "Congrats! Tumor NOT detected"}, status=200)
    else:
        return JsonResponse({'success': False, 'error': 'No image file found'}, status=400)
