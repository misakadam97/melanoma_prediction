import numpy as np
import os    
os.environ['THEANO_FLAGS'] = "device=cpu,floatX=float32"    
import theano
import keras
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
#%%loadthe data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#%%divide the data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255	
#%%Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
#%%create a model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu',\
                 data_format="channels_first", input_shape=X_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#%%Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%train  model
model.fit(X_train, Y_train, 
          batch_size=32, epochs=10, verbose=1)

#%%score model	
score = model.evaluate(X_test, Y_test, verbose=2)
