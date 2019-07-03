#Convolutional NNs

#Here, it wouldn't make sense to add the independent variable into tupples with the observations;

#Keras: we only need to prepare a very special structure for the dataset
#- separate into training set folder and test set folder
#- separate into two main folders for the two main types of data - cats and dogs
#- we are only using a subset of Kaggle's dataset that contains 25 000 images in each category
#- our preprocessing is basically done (manually), we only have to do the feature scaling (before fitting the CNN to our data)



#Part 1 - Building the CNN
#
#Importing the Keras libraries and packages:
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Initialising the CNN: the same as an ANN
classifier = Sequential()

#Step 1 - Convolution (not a fully conected layer like we had before)
#the images will be later converted to have the same format
#Note that the order in the input_shape argument is different for TensorFlow (the default one is Theano!)
classifier.add(Convolution2D(32, (3, 3), input_shape=(128 , 128, 3), activation = 'relu'))

#Step 2 - Pooling (2,2) is standard
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 1*: add another Convolutional layer, apply on the pooled feature maps obtained in the previous step
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
#Step 2*: add another pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Again:
#Step 1*: add another Convolutional layer, apply on the pooled feature maps obtained in the previous step
classifier.add(Convolution2D(128, (3, 3), activation = 'relu'))
#Step 2*: add another pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Again:
#Step 1*: add another Convolutional layer, apply on the pooled feature maps obtained in the previous step
classifier.add(Convolution2D(128, (3, 3), activation = 'relu'))
#Step 2*: add another pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

##Again:
##Step 1*: add another Convolutional layer, apply on the pooled feature maps obtained in the previous step
#classifier.add(Convolution2D(128, (3, 3), activation = 'relu'))
##Step 2*: add another pooling layer
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening
# By applying steps 1,2 we get information about how a given pixel is spatially connected
# to its surrounding pixes - that's why we don't feed the image to the ANN directly.
# Also, the spatial info is not lost with flattening, because it's been already encoded in steps 1,2.
classifier.add(Flatten())

#Step 4 - Create the classic ANN (Full connection)
#hidden layer (number chosen for experimenting and by experience, also best is a power of 2)
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(rate=0.4))
#output layer: for binary outcome: sigmoid; outcome with more categories: softmax
classifier.add(Dense(units=64, kernel_initializer="uniform", activation="relu"))
classifier.add(Dropout(rate=0.4))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the dataset
#Image augmentation: preprocessing the images to avoid overfitting
#We are using Keras documentation (online) -> Preprocessing -> Image
#It will create many batches of our images and apply random transformations to them (rotation, flipping, shifting, shearing)
#This effectively increases the number of training images and we shouldn't find the same image in different batches!

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#rescale pixes of images from the test set so that they have values between 0 and 1
test_datagen = ImageDataGenerator(rescale=1./255)

#creates the training set, divides into batches
#Carefull that we are in the correct working folder!
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

#creates the test set.
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

#fits the CNN to the training set while testing the performance on the test set.
#steps_per_epoch should be the number of images; validation_steps = test set dimension
classifier.fit_generator(
        training_set,
        steps_per_epoch=(8000/32),
        epochs=50,
        validation_data=test_set,
        validation_steps=(2000/32))

#Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/DSC_0123.jpg',
                            target_size = (64, 64))
#next, we need to make the image have the desired shape. Currently, it has 3 layers (RGB)
test_image = image.img_to_array(test_image)
#add an extra dimention to our 3D array corresponding to the batch (even for 1 input only!).
#We add the extra dimension as a first column:
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)
#What is the mapping between cats, dogs, and the values of "result"?
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'











