import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k
import sklearn.metrics
from scipy.ndimage import gaussian_filter

(x_train, y_train) = (np.load('data/train_X.npy', mmap_mode='r'), np.load('data/train_Y.npy', mmap_mode='r'))
(x_test, y_test) = (np.load('data/test_X.npy', mmap_mode='r'), np.load('data/test_Y.npy', mmap_mode='r'))
print(y_test, x_test)
sigma_1 = 6 #y_ish
sigma_2 = 0.4 #x_ish


img_rows, img_cols = 28, 28

if k.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	inpx = (1, img_rows, img_cols)

else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	inpx = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_test)
print(x_test.shape)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

inpx = Input(shape=inpx)
layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx)
layer2 = MaxPooling2D(pool_size=(3, 3))(layer1)
layer3 = Conv2D(64, (3, 3), activation='relu')(layer2)
layer5 = Flatten()(layer3)
layer6 = Dense(250, activation='relu')(layer5)
layer7 = Dense(70, activation="relu")(layer6)
layer8 = Dense(4, activation="linear")(layer7)
layer9 = Dense(5, activation='softmax')(layer8)

model = Model([inpx], layer9)
model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1, batch_size=256)

score = model.evaluate(x_test, y_test, verbose=0)
print('loss=', score[0])
print('accuracy=', score[1])