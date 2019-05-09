import pandas as pd

predictors = pd.read_csv('inputs-breast.csv')
diagnosis = pd.read_csv('outputs-breast.csv')

from sklearn.model_selection import train_test_split
# Splits the dataset between testing and training where the training collection represents 75% of the set 
predictors_training, predictors_test, training_diagnosis, testing_diagnosis = train_test_split(predictors, diagnosis, test_size=0.25) 

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
# Builds the first layer of our neural network
classifier.add(Dense(units=20, activation='relu',
                     kernel_initializer='random_uniform',
                     input_dim=30,
                     use_bias=True))
# Builds a hidden layer with this configuration
classifier.add(Dense(units=20, activation='relu',
                     kernel_initializer='random_uniform',
                     use_bias=True))
# Builds up a custom optimizer with the setup below
optimizer = keras.optimizers.Adadelta(lr=1.0,rho=0.95, epsilon=None, decay=0.001)
# Adds a output layer
classifier.add(Dense(units=1, activation='sigmoid'))
# Compiles the network adding up the loss function, the customm optimizer and the metrics to we evaluate
classifier.compile(optimizer=optimizer,loss='binary_crossentropy')

# Fits up the data to the network
classifier.fit(predictors_training, training_diagnosis, batch_size=4, epochs=100)

# Evaluation:
predictions = classifier.predict(predictors_test)
predictions = (predictions > 0.5) # format to boolean values

from sklearn.metrics import accuracy_score
precision = accuracy_score(testing_diagnosis, predictions) # This evaluates the precision of our model (over the training data)
