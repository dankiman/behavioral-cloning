import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def read_image(image_path):
    '''
    Read in an RGB image from
    :param image_path: The path to the image file
    :return: RGB image array
    '''
    if not os.path.exists(image_path):
        print('Path does not exist: {0}'.format(image_path))
    return mpimg.imread(image_path)

def resize_image(image):
    '''
    Resize and crop an image to specific dimensions.

    :param image: Image array
    :return: resized image
    '''
    # Resize and Crop image to (16, 32, 3)
    dim = (32, int(image.shape[0] * (32.0 / image.shape[1])))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def stream_process(image):
    '''
    Process streaming images

    :param image: Image array
    :return: processed image
    '''
    image = resize_image(image)
    return image

def preprocess(image_path):
    '''
    Process image for training data

    :param image_path: Path to image file
    :return: processed image
    '''
    image_path = image_path.strip()
    # For relative pathing with Udacity data
    image_path = os.path.join('/Users/JimWinquist/Desktop/data', image_path)
    image = read_image(image_path)
    image = resize_image(image)
    return image