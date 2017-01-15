import os
import json
import cv2
import argparse
import numpy as np
import pandas as pd
import image_process

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ELU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
print('Modules imported.')

#DRIVING_LOG = "/Users/JimWinquist/Desktop/cloning_data_clean/driving_log.csv"
DRIVING_LOG = "/Users/JimWinquist/Desktop/data/driving_log.csv"
#LEFT_DRIVING = "/Users/JimWinquist/Desktop/left_driving/driving_log.csv"
#RIGHT_DRIVING = "/Users/JimWinquist/Desktop/right_driving/driving_log.csv"
SPOT_TRAINING = "/Users/JimWinquist/Desktop/spot_training_after_bridge/driving_log.csv"
JSON_PATH = "model.json"
WEIGHTS_PATH = "model.h5"

def load_driving_data(driving_log):
    '''
    Load driving data into a pandas dataframe for further processing

    :param driving_log: csv file containing raw driving data
    :return: pandas dataframe of driving data
    '''
    df = pd.read_csv(driving_log, header=None, names=['center_image', 'left_image',
                                                      'right_image', 'steering_angle',
                                                      'throttle', 'break', 'speed'])
    return df

def get_model():
    '''
    Create keras model architecture for training driving behavior.

    :return: model
    '''
    # Simplified model
    row, col, ch = 16, 32, 3  # image dimensions

    model = Sequential()
    model.add(Lambda(lambda x: x/128. - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 3, 3, init='he_normal', activation='relu', border_mode="valid"))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4,4), border_mode='valid'))
    model.add(Dropout(.2))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), "mse")

    return model

def batch_generator(driving_data, batch_size=8):
    '''
    Generate batches of images to be passed to the model.

    :param driving_data: pandas dataframe containing all of the driving data
    :param batch_size: the number of images in each batch
    :return: a batch of image(features) and steering angle(labels)
    '''
    num_rows = driving_data.shape[0]
    # Initialize array with dimensions (batch_size, row, col, ch)
    X_train = np.zeros((batch_size, 16, 32, 3))
    y_train = np.zeros(batch_size)
    index = None
    while True:
        for i in range(0, batch_size, 4):
            if index is None or index >= num_rows:
                index = 0
            # Add center camera image and steering label
            angle = driving_data.iloc[index].steering_angle
            X_train[i] = image_process.preprocess(driving_data['center_image'].iloc[index])
            y_train[i] = angle
            # Add left camera image with +0.25 steering angle offset
            X_train[i+1] = image_process.preprocess(driving_data['left_image'].iloc[index])
            y_train[i+1] = angle + 0.25
            # Add right camera image with -0.25 steering angle offset
            X_train[i+2] = image_process.preprocess(driving_data['right_image'].iloc[index])
            y_train[i+2] = angle - 0.25
            # Add flipped center camera image with negated steering angle
            X_train[i+3] = cv2.flip(image_process.preprocess(driving_data['center_image'].iloc[index]), 1)
            y_train[i+3] = -1 * angle
            index += 1
        yield (X_train, y_train)

def main():
    angle_threshold = 0.001
    if os.path.exists(JSON_PATH):
        # Reload Model and Weights and train with new data and lower learning rate
        train = load_driving_data(DRIVING_LOG)
        train = train[abs(train['steering_angle']) > angle_threshold]

        with open(JSON_PATH, 'r') as jfile:
            model = model_from_json(json.load(jfile))

        model.compile(Adam(lr=0.000001), "mse")
        model.load_weights(WEIGHTS_PATH)
        history = model.fit_generator(batch_generator(train, batch_size=8),
                                      samples_per_epoch=train.shape[0]*3,
                                      nb_epoch=3, verbose=1, callbacks=[],
                                      validation_data=None, nb_val_samples=None,
                                      class_weight=None, max_q_size=10,
                                      nb_worker=1, pickle_safe=False)
    else:
        # Load raw driving data
        print('Loading Data...')
        driving_data = load_driving_data(DRIVING_LOG)

        # Extract only steering angles above some threshold
        train = driving_data[abs(driving_data['steering_angle']) > angle_threshold]

        print('Building Model...')
        model = get_model()

        print('Training Model...')
        history = model.fit_generator(batch_generator(train, batch_size=8),
                                      samples_per_epoch=train.shape[0]*3,
                                      nb_epoch=5, verbose=1, callbacks=[],
                                      validation_data=None, nb_val_samples=None,
                                      class_weight=None, max_q_size=10,
                                      nb_worker=1, pickle_safe=False)

    print('Saving model...')
    with open(JSON_PATH, 'w') as f:
        json.dump(model.to_json(), f)

    model.save_weights(WEIGHTS_PATH)

if __name__ == '__main__':
    print('Running model.py')
    main()
    print('Model Training Complete.')
