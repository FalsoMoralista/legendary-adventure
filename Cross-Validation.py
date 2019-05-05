import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

predictors = pd.read_csv('/home/luciano/ic/lectures/breast_cancer/datasets/entradas-breast.csv')
classes = pd.read_csv('/home/luciano/ic/lectures/breast_cancer/datasets/saidas-breast.csv')


# Cross validation is the technique used in order to try to prevent/reduce overfitting
# by permutating the traning set over iterations.
#
# Creates the neural network (DFF) based on the previous configuration
def arise():
    classifier = Sequential()

    classifier.add(Dense(units = 20,# Adds the Input layer 
                         activation='relu', # Activation function
                         kernel_initializer='random_uniform', 
                         input_dim=30))# Number of inputs     
    
    classifier.add(Dropout(0.25)) # Prevents overfitting through randomly setting a fraction rate of input units to 0        
    
    classifier.add(Dense(units = 20,# " "  Hidden layer
                         activation='relu', # " " 
                         kernel_initializer='random_uniform'))         
  
    classifier.add(Dropout(0.25)) # Prevents overfitting through randomly setting a fraction rate of input units to 0 

    classifier.add(Dense(units = 1, activation='sigmoid')) # " " Output layer

    optmizer = keras.optimizers.adadelta(lr=1, rho=0.95,epsilon=None, decay=0.001) # Sets up a custom optimizer
    classifier.compile(optimizer=optmizer, # Compiles with the loss function and the used metric
                       loss='binary_crossentropy',metrics=['binary_accuracy']) 
    return classifier


dff = KerasClassifier(build_fn=arise, epochs=100, batch_size=4) # Initializes

experiment = cross_val_score(estimator=dff, # Runs the experiment using cross validation
                             X=predictors,
                             y=classes, 
                             cv=10, # Determinate the cross validation "iterations"
                             scoring='accuracy')

mean = experiment.mean() # Mean of outputed values from the neural network
sd = experiment.std() # Standard deviation (measures how the network is fitting )

classifier = dff.build_fn().to_json() # Format to json
with open('data/breast/exported/exported_classifier.json','w') as json_file:
    json_file.write(classifier) # Exports it to a folder
classifier = dff.build_fn().save_weights('data/breast/exported/exported_classifier.h5') # Do same to the weights