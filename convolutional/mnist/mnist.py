import matplotlib.pyplot as mplib
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

(X_training, y_training), (X_test_y_test) = mnist.load_data() # Loads up the data from the mnist dataset splitting between trraining and testing
mplib.imshow(X_training[0], cmap = 'gray') # Show an image on the console, changing its scale to gray
mplib.title('Classe ' + str(y_training[0])) # Show an image's class