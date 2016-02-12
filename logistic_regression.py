# Built-in python libs
import numpy as np
import gzip
import cPickle as pickle

# Load keras modules
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

# load and preprocess dataset
(X_train, y_train), (X_test, y_test) = pickle.load(gzip.open('usps.pkl.gz','rb'))


# -- the input must be in the form of n x d matrix (n: #data, d: dimensionality)
[n_train, _, d1, d2] = X_train.shape
d = d1*d2
X_train = np.reshape(X_train, (n_train, d)).astype('float32')
X_train /= 255.0

[n_test, _, d1, d2] = X_test.shape
d = d1*d2
X_test = np.reshape(X_test, (n_test, d)).astype('float32')
X_test /=255.0

# -- the label must be converted into one-hot vector
Y_train = np_utils.to_categorical(y_train, nb_classes=10)
Y_test = np_utils.to_categorical(y_test, nb_classes=10)


# Create logistic regression model
model = Sequential()
model.add(Dense(10, input_dim=256))
model.add(Activation('softmax'))


# Define optimizer and loss function
opt = SGD(lr=1e-2, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt)


model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
	batch_size=10, nb_epoch=20, show_accuracy=True)