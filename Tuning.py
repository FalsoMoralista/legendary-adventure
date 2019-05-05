import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import os
from google.colab import drive
drive.
os.listdir()


predictors = pd.read_csv('/home/luciano/ic/lectures/breast_cancer/datasets/entradas-breast.csv')
classes = pd.read_csv('/home/luciano/ic/lectures/breast_cancer/datasets/saidas-breast.csv')
    
optmizer = keras.optimizers.adadelta(lr=1, rho=0.95,epsilon=None, decay=0.001) # Sets up a custom optimizer

# Creates the neural network (DFF) based on the parametrized configuration
def arise(optimizer, loss, kernel_initializer, activation, neurons):
    classifier = Sequential()

    classifier.add(Dense(units = neurons,# Adds the Input layer 
                         activation=activation, # Activation function
                         kernel_initializer=kernel_initializer, 
                         input_dim=30))# Number of inputs     
    
    classifier.add(Dropout(0.25)) # Prevents overfitting through randomly setting a fraction rate of input units to 0        
    
    classifier.add(Dense(units = neurons,# " "  Hidden layer
                         activation=activation, # " " 
                         kernel_initializer=kernel_initializer))         
  
    classifier.add(Dropout(0.25)) # Prevents overfitting through randomly setting a fraction rate of input units to 0 

    classifier.add(Dense(units = 1, activation='sigmoid')) # " " Output layer
    
    classifier.compile(optimizer=optmizer, # Compiles with the loss function and the used metric
                       loss=loss,metrics=['binary_accuracy']) 
    return classifier


dff = KerasClassifier(build_fn=arise) # Initializes

parameters = {
        'batch_size': [4,10,30],
        'epochs':  [100,120],
        'optimizer': ['adam','adadelta'],
        'loss': ['binary_crossentropy','hinge'],
        'kernel_initializer': ['random_uniform', 'normal'],
        'activation':['relu','tanh'],
        'neurons': [20,12,6],
        }

grid_search = GridSearchCV(estimator=dff,
                           param_grid= parameters,
                           scoring='accuracy',
                           cv=5)
grid_search= grid_search.fit(predictors,classes)
best_parameters=grid_search.best_params_
best_accuracy= grid_search.best_score_

