import pandas as pd

predictors = pd.read_csv('entradas-breast.csv')
diagnosis = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split
# Splits the dataset between testing and training where the training collection represents 75% of the set 
predictors_training, predictors_test, training_diagnosis, testing_diagnosis = train_test_split(predictors, diagnosis, test_size=0.25) 

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
# Builds the hidden layer of our neural network, by default initializes the input layer as well
classifier.add(Dense(units=20, activation='relu',
                     kernel_initializer='random_uniform', input_dim= 30, use_bias=True))
# Builds another hidden layer with this configuration
classifier.add(Dense(units=20, activation='relu',
                     kernel_initializer='random_uniform', use_bias=True))
# Builds up a custom optimizer with the setup below
optimizer = keras.optimizers.Adadelta(lr=1.0,rho=0.95, epsilon=None, decay=0.001)
# Adds a output layer with the activation function that will get us values between 0 and 1
classifier.add(Dense(units=1, activation='sigmoid'))
# Compiles the network adding up the loss function, the customm optimizer and the metrics to we evaluate
classifier.compile(optimizer=optimizer,loss='binary_crossentropy', )

#diagnosisifier.compile(optimizer='adadelta',loss='binary_crossentropy', metrics=['binary_accuracy'])
# Fits up the data to the network
classifier.fit(predictors_training, training_diagnosis, batch_size=4, epochs=100)
# Evaluation:
predicoes = classifier.predict(predictors_test)
predicoes = (predicoes > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(testing_diagnosis, predicoes)
matrix = confusion_matrix(testing_diagnosis, predicoes)
#result = diagnosisifier.evaluate(predictors_test,testing_diagnosis)
####################################################################
weights0 = classifier.layers[0].get_weights()
weights1 = classifier.layers[1].get_weights()

