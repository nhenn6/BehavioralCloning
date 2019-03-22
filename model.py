import os
import random
from random import shuffle
import csv
import cv2
import numpy as np
import keras



# all steps to preprocess the image; defined own function to leave room for expansion if need be
def preprocess_image(image):
    # all steps to preprocess the image; defined own function to leave room for expansion if need be
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    return image

# all steps to augment the image; defined own function to leave room for expansion if need be
def augment_image(image, steering_angle):
    #apply gaussian blur to all images
    rand_int = random.randint(0, 4)
    ksize = [ 1, 3, 5, 7, 9]
    cv2.GaussianBlur(image,(ksize[rand_int],ksize[rand_int]),0)
    
    return image, steering_angle

#model was trained on Eric Lavigne's "Tiny data set" from slack. 
#Set consists of 25 image sets (right, left, center) from the first track and a few from the other tracks.
lines = []
with open('ericlavigne-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'ericlavigne-data/IMG/' + filename
        image = cv2.imread(current_path)
        image = preprocess_image(image)
        images.append(image)
    correction = 0.1
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)
    
    
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    image, measurement = augment_image(image, measurement)
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    #flip every image to create a balanced data set
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)
    

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D


#using the nvidia architechture
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
#crop the image to remove the background and focus on the track
model.add(Cropping2D(cropping=((60,20), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#add in dropout to combat overfitting
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)

model.save('model.h5')
