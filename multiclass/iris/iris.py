import keras
import pandas as pd 
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

dataset = pd.read_csv('../../data/iris/iris.csv')

# Splits the dataframe between predictor attributes and classes
predictors = dataset.iloc[:, 0:4].values
classes = dataset.iloc[:, 4].values

labelencoder = LabelEncoder()

classes = labelencoder.fit_transform(classes) # Encodes categorical values
classes_dummy = np_utils.to_categorical(classes)

# By default builds a fully connected neural network (FeedForward - FF) that 
# has as inputs 4 attributes (Sepal and Petal -  width/length), that being 4 neurons
# on the input layer, and 3 hidden layers with 3 neurons on each of classes.
# --The configuration used on the hidden and input layer is the activation function 'relu' &
# the initializer function was set to the default function ('normal'). It was also added
# a dropout rate of 33% between each layer.
# --The output function was set to softmax cause it gives as result the probability of 
# a sample being from a specific class.   
def build():
    classifier = Sequential()
    classifier.add(Dense(units=6,
                         activation='relu',
                         kernel_initializer='random_uniform',
                         input_dim=4))
    classifier.add(Dropout(0.125))
    classifier.add(Dense(units = 6,# " "  Hidden layer
                         activation='relu', # " " 
                         kernel_initializer='random_uniform'))         
    classifier.add(Dropout(0.125)) # Prevents overfitting through randomly setting a fraction rate of input units to 0 
    classifier.add(Dense(units = 6,# " "  Hidden layer
                         activation='relu', # " " 
                         kernel_initializer='random_uniform'))         
    classifier.add(Dropout(0.125)) # Prevents overfitting through randomly setting a fraction rate of input units to 0 
    classifier.add(Dense(units = 3, activation='softmax')) # " " Output layer
    optmizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    classifier.compile(optimizer=optmizer, # Compiles with the loss function and the used metric
                       loss='categorical_crossentropy',
                       metrics=['categorical_accuracy']) 
    return classifier

dff = KerasClassifier(build_fn=build, epochs=1000, batch_size=6) # Initializes
experiment = cross_val_score(estimator=dff, # Runs the experiment using cross validation
                             X=predictors,
                             y=classes, 
                             cv=4, # Determinate the cross validation "iterations"
                             scoring='accuracy')

mean = experiment.mean() # Mean of outputed values from the neural network
sd = experiment.std() # Standard deviation (measures how the network is fitting )