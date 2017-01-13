import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def read_image(image_path):
    if not os.path.exists(image_path):
        print('Path does not exist: {0}'.format(image_path))
    return mpimg.imread(image_path)

def convert_image(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def resize_crop_image(image):
    # Resize and Crop image to (25, 80, 3)
    crop = image.copy()
    crop = crop[40:140,:]
    dim = (80, int(crop.shape[0] * (80.0 / crop.shape[1])))
    return cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)

def stream_process(image):
    image = convert_image(image)
    image = resize_crop_image(image)
    return image

def preprocess(image_path):
    image_path = image_path.strip()
    image = read_image(image_path)
    image = convert_image(image)
    image = resize_crop_image(image)
    return image