import numpy as np
np.random.seed(123)
from keras.models import Sequential # Linear neural network layers
from keras.layers import Dense,  Dropout, Activation, Flatten # Core layers
from keras.layers import Convolution2D as Conv2D, MaxPooling2D as MaxP2D, MaxPooling3D as MaxP3D # CNN layers
from keras.utils import np_utils # Utils for transforming data

from keras.datasets import cifar10 # Image datasets
from matplotlib import pyplot as plt

# Load pre-shuffled cifar10 data into train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#my_utils.display_image_data(x_train[0])
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print(x_train.shape)
print(y_train.shape)
print(y_test[0])

model = Sequential()
# filters, kernel_size, strides, input_shape = (width, height, depth)
conv_layer = Conv2D(32, 3, 3, activation='relu', input_shape=(32, 32, 3))
model.add(conv_layer)
model.add(Conv2D(32, 3, 3, activation='relu'))

model.add(Conv2D(32, 3, 3, activation='relu')) # Added
model.add(Conv2D(32, 3, 3, activation='relu')) # Added

model.add(MaxP2D(pool_size=(2, 2)))
# Regularizing to prevent overfitting
# Creates a different network and use mean averaging
model.add(Dropout(0.25))

model.add(Flatten()) # Weights from conv layers has to be made 1-dimensional (flattened) before connected to dense layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Set callback to set early-stopping rules
model.fit(x_train, y_train, batch_size=32, nb_epoch=5, verbose=1)

#model.save('image_model.h5')

# Evaluate model on test data
#score = model.evaluate(x_test, y_test, verbose=0)
