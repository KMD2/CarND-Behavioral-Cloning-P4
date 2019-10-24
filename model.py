# Imports
import csv
import cv2
import numpy as np
import sklearn
import math

## Setup Keras
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Function for importing the data from driving_log.csv
def reading_data (path, angle_correction=0.2):
    images_names=[] #data
    angles=[] #labels
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for row in reader:
            for i in range (3):
                source_path = row[i]
                filename = source_path.split('/')[-1]
                current_path = './data/IMG/' + filename

                images_names.append(current_path)

                # Read the steering angle
                angle = float(row[3])

                # Add correction to the steering angle for the left and right images
                if i==0:
                    angles.append(angle)
                elif i == 1:
                    angles.append(angle + angle_correction)
                else:
                    angles.append(angle - angle_correction)
                    
    images_names = np.array(images_names)
    angles = np.array(angles)
    
    return images_names, angles

# Generators function to be more memory_efficient
def generator(samples, labels, batch_size=32):
    num_samples = len(labels)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples, labels)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]

            images = []
            angles = []
              
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                
                # Augment data by flipping the image
                image_flipped = np.fliplr(image)
                images.append(image_flipped)
                
            for batch_label in batch_labels:
                angles.append(batch_label)
                
                # Find the label for the flipped image
                label_flipped = -batch_label
                angles.append(label_flipped)
                

            # trim image to only see section with road
            X = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(X, y)
            
# CNN architecture - NVIDIA

def nvidia_model():
    ## Create a model instance
    model = Sequential()

    ## Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    ## Crop Images
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=    (160,320,3))) 

    model.add(Conv2D(24, (5, 5), subsample = (2,2) ,activation='relu'))
    model.add(Conv2D(36, (5, 5), subsample = (2,2) ,activation='relu'))
    model.add(Conv2D(48, (5, 5), subsample = (2,2) ,activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    return model


# Data pre-processing     
image_names, angles = reading_data(path = './data/driving_log.csv', angle_correction = 0.2)
train_names, valid_names, train_angles, valid_angles = train_test_split(image_names,angles, test_size=0.2)

print('Number of training images = ', len(train_names))
print('Number of validation images = ', len(valid_names))


# Compile and train the model using the generator function
batch_size = 32
train_generator = generator(train_names, train_angles,batch_size = batch_size)
valid_generator = generator(valid_names, valid_angles,batch_size = batch_size)

## Create the model

model = nvidia_model()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_names)/ batch_size), validation_data=valid_generator,validation_steps=math.ceil(len(valid_names)/batch_size),epochs=3, verbose=1)

model.save('model.h5')
    
## print the keys contained in the history object
print(history_object.history.values())

np.savez('loss_history', loss=np.array(history_object.history['loss']), val_loss=np.array(history_object.history['val_loss']))
f = np.load('loss_history.npz')

## plot the training and validation loss for each epoch
plt.plot(f['loss'], label="training set")
plt.plot(f['val_loss'], label="validation set")
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig("examples/loss.jpg", bbox_inches='tight')
#plt.show()



model.summary()
        
        
        
                
                
        

        
        
        
        
    