"Wednesday Model, 25th March 2020"
#image classification

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow  import keras
#from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

import os
import numpy as np
import matplotlib.pyplot as plt
import glob 

PATH= 'D:\\BE PROJ\\leaf disease detection using image processing block diagram_files\\new'


train_dir = os.path.join(PATH, 'D:\\BE PROJ\\leaf disease detection using image processing block diagram_files\\new\\train1')
validation_dir = os.path.join(PATH, 'D:\\BE PROJ\\leaf disease detection using image processing block diagram_files\\new\\val1')
test_dir = os.path.join(PATH, 'D:\\BE PROJ\\leaf disease detection using image processing block diagram_files\\new\\test1') 
                   

train_blb_dir = os.path.join(train_dir, 'bacterial leaf blight')  # directory with our training leaf blight pictures
train_bs_dir = os.path.join(train_dir, 'brown spot')  # directory with our training brown spot pictures
#train_health_dir = os.path.join(train_dir, 'Healthy')  # directory with our training healthy rice pictures
train_ls_dir = os.path.join(train_dir, 'leaf smut')  # directory with our training leaf smut pictures

test_blb_dir = os.path.join(test_dir, 'bacterial leaf blight')  # directory with our test leaf blight pictures
test_bs_dir = os.path.join(test_dir, 'brown spot')  # directory with our test brown spot pictures
#test_health_dir = os.path.join(test_dir, 'Healthy')  # directory with our test healthy rice pictures
test_ls_dir = os.path.join(test_dir, 'leaf smut')  # directory with our test leaf smut pictures


validation_blb_dir = os.path.join(validation_dir, 'bacterial leaf blight')  # directory with our validation leaf blight pictures
validation_bs_dir = os.path.join(validation_dir, 'brown spot')  # directory with our validation brown spot pictures
#validation_health_dir = os.path.join(validation_dir, 'Healthy')  # directory with our validation healthy rice pictures
validation_ls_dir = os.path.join(validation_dir, 'leaf smut')  # directory with our validation leaf smut pictures

num_blb_tr = len(os.listdir(train_blb_dir))
num_bs_tr = len(os.listdir(train_bs_dir))
#num_health_tr = len(os.listdir(train_health_dir))
num_ls_tr = len(os.listdir(train_ls_dir))

num_blb_ts = len(os.listdir(test_blb_dir))
num_bs_ts = len(os.listdir(test_bs_dir))
#num_health_ts = len(os.listdir(test_health_dir))
num_ls_ts = len(os.listdir(test_ls_dir))

num_blb_val = len(os.listdir(validation_blb_dir))
num_bs_val = len(os.listdir(validation_bs_dir))
#num_health_val = len(os.listdir(validation_health_dir))
num_ls_val = len(os.listdir(validation_ls_dir))

total_train = num_blb_tr + num_bs_tr + num_ls_tr
total_test = num_blb_ts + num_bs_ts + num_ls_ts
total_val = num_blb_val + num_bs_val + num_ls_val
num_classes=len(glob.glob(train_dir+"/*"))  

print('total training bacterial leaf blight images:', num_blb_tr)
print('total training brown spot images:', num_bs_tr)
#print('total training healthy images:', num_health_tr)
print('total training leaf smut images:', num_ls_tr)

print('total testing bacterial leaf blight images:', num_blb_ts)
print('total testing brown spot images:', num_bs_ts)
#print('total testing healthy images:', num_health_ts)
print('total testing leaf smut images:', num_ls_ts)

print('total validation bacterial leaf blight images:', num_blb_val)
print('total validation brown spot images:', num_bs_val)
#print('total validation healthy images:', num_health_val)
print('total validation leaf smut images:', num_ls_val)
print("*************************")

print("Total training images:", total_train)
print("Total test images:", total_test)
print("Total validation images:", total_val)
print("Classes", num_classes)

batch_size = 64
epochs = 10
IMG_HEIGHT = 256
IMG_WIDTH = 256
input_shape=(IMG_WIDTH,IMG_HEIGHT,3)
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our test data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

#Visualize training images 
sample_training_images, _ = next(train_data_gen)
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(sample_training_images[:5])

#Initializing the cnn,creating an object od Sequential
classifier = Sequential()
classifier.add(Conv2D(32, (5, 5),input_shape=input_shape,activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(64, (3, 3),activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(128, (3, 3),activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2))) 
classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu', activity_regularizer=regularizers.l1(0.01)))
classifier.add(Dropout(0.5))          
classifier.add(Dense(num_classes,activation='softmax'))

opt=keras.optimizers.Adam(lr=0.001, decay=0.001)
classifier.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy']) 


#View all the layers of the network using the model's summary method:
classifier.summary()

#this function fits the cnn created above to the images
history = classifier.fit_generator(train_data_gen,
                                   steps_per_epoch=total_train//batch_size,
                                   epochs=epochs,
                                   validation_data=val_data_gen,
                                   validation_steps=total_val//batch_size)

#visualizing the acccuracy and loss of the model with graphs 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#save model
classifier.save('classifier.h5')
                      
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
