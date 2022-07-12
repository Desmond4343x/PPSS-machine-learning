from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
np.random.seed(10)

from time import time
import numpy as np
import keras.backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import sklearn.metrics



#Loading training data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))
print(X_train.shape)
print(X_test.shape)

#Neural net
input_img = Input(shape=(784,))
#cosh

encoded = Dense(units=628, activation='relu')(input_img)
encoded = Dense(units=180, activation='relu')(encoded)
encoded = Dense(units=66, activation='relu')(encoded)
encoded = Dense(units=19, activation='relu')(encoded)
encoded = Dense(units=8, activation='relu')(encoded)
encoded = Dense(units=4, activation='relu')(encoded)

decoded = Dense(units=8, activation='relu')(encoded)
decoded = Dense(units=19, activation='relu')(decoded)
decoded = Dense(units=66, activation='relu')(decoded)
decoded = Dense(units=180, activation='relu')(decoded)
decoded = Dense(units=628, activation='relu')(decoded)
decoded = Dense(units=784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

autoencoder.summary()

encoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(X_train, X_train,
                epochs=400,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))


#Prepeares the images for plotting
encoded_imgs = encoder.predict(X_test)
predicted = autoencoder.predict(X_test)

#Plots the figueres
plt.figure(figsize=(40, 4))
for i in range(10):
	# display original images
	ax = plt.subplot(3, 20, i + 1)
	plt.imshow(X_test[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# display encoded images
	ax = plt.subplot(3, 20, i + 1 + 20)
	plt.imshow(encoded_imgs[i].reshape(2, 2))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# display reconstructed images
	ax = plt.subplot(3, 20, 2 * 20 + i + 1)
	plt.imshow(predicted[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

#Plots the clusters for the encoded data
def plot_label_clusters(encoded, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z = encoder.predict(data.reshape(len(data),784))
    print(z.shape)
    plt.figure(figsize=(12, 10))
    plt.scatter(z[:, 0], z[:, 1], c=labels, cmap="RdYlBu_r")
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


plot_label_clusters(encoded, X_test, Y_test)



plt.show()
