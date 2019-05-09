import pandas as pd
import os
print(os.getcwd())

df = pd.read_csv('air_quality_Nov2017.csv')
df['Station'].value_counts()
# Drops not used attribute
df = df.drop('DateTime',axis=1) 

# Drops all cells that contais '--' value for the attribute 'AirQuality'
df = df[df.AirQuality != '--']

# Drop all 'na'(not a number) values.
# --The best solution may would be list the non null values and fill the null
# values according to the probability of each non null value instead of dropping 
# half of the dataset, though it may be more arduous. TODO LATER MAYBE
df = df.dropna()

# Splits the attributes that will be used to train the network to predict the air quality
# from the real measured values, that will be used to calculate the error.
real_air_quality = df.iloc[:,1].values
predictors = df.iloc[:,2:14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Now we need to encode the categorical attributes so that the algorithm takes real input values 
# instead of Good/Moderate, for example.
labelEncoder1 = LabelEncoder()
real_air_quality[:] = labelEncoder1.fit_transform(real_air_quality[:]) # Encodes using the label encoder

# Now we will encode the other cathegorical attributes from our dataframe, like
# latitude, longitude, date, O3 Hour, etc...
# [0,1,2,3,5,6,8,9,11] - These columns need to be encoded with the label & hot encoder. 
labelEncoder2 = LabelEncoder()

predictors[:,0] = labelEncoder2.fit_transform(predictors[:,0])
predictors[:,1] = labelEncoder2.fit_transform(predictors[:,1])
predictors[:,2] = labelEncoder2.fit_transform(predictors[:,2])
predictors[:,3] = labelEncoder2.fit_transform(predictors[:,3])
predictors[:,5] = labelEncoder2.fit_transform(predictors[:,5])
predictors[:,6] = labelEncoder2.fit_transform(predictors[:,6])
predictors[:,8] = labelEncoder2.fit_transform(predictors[:,8])
predictors[:,9] = labelEncoder2.fit_transform(predictors[:,9])
predictors[:,11] = labelEncoder2.fit_transform(predictors[:,11])

# After that all cathegorical attributes were parsed into numeric values (eg., Good = 1, Moderate = 0)
# though we need to turn them into dummy values so our algorithm 
# doesn't consider that good is bigger than moderate, cause that wouldn't make
# sense.
hotEncoder = OneHotEncoder(categorical_features=[0,1,2,3,5,6,8,9,11])
predictors = hotEncoder.fit_transform(predictors).toarray()

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score


# Builds the feed forward neural network 
# -- A question that came up on my mind was how the model will be able to predict 
# the air quality for a given station since we are taking as input data from 5 
# different stations and outputing a unique value. Then i remembered that 
# we used geolocation as attributes to train the model, so when the put as input 
# the data for a single station it should predict correctly, cause the model
# was trained to use the location as attribute to predict the air quality.--
def build():
    regressor = Sequential()
    regressor.add(Dense(units=403, activation='relu', input_dim=806)) # Input/first layer
    regressor.add(Dropout(0.125))
    regressor.add(Dense(units=403, activation='relu')) # Second layer
    regressor.add(Dropout(0.125))
    regressor.add(Dense(units=1,activation='linear')) # Output
    regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics = ['mean_absolute_error'])
    return regressor

regressor = KerasRegressor(build_fn=build, epochs=1000, batch_size=806)
results= cross_val_score(estimator=regressor, X = predictors, y = real_air_quality, cv=4, scoring = 'neg_median_absolute_error')

mean=results.mean()
sd = results.std()
print(abs(mean))

#test_station = df.iloc[73,2:14].values
#test_station = test_station.reshape(-1,1)
#test_station[:] = labelEncoder3.fit_transform(test_station[:])
#labelEncoder3 = LabelEncoder()
#test_station[0] = labelEncoder3.fit_transform(test_station[0])
#test_station[1] = labelEncoder3.fit_transform(test_station[1])
#test_station[2] = labelEncoder3.fit_transform(test_station[2])
#test_station[3] = labelEncoder3.fit_transform(test_station[3])
#test_station[5] = labelEncoder3.fit_transform(test_station[5])
#test_station[6] = labelEncoder3.fit_transform(test_station[6])
#test_station[8] = labelEncoder3.fit_transform(test_station[8])
#test_station[9] = labelEncoder3.fit_transform(test_station[9])
#test_station[11] = labelEncoder3.fit_transform(test_station[11])

# Now let us test with a instance of a station 

# Load Models
from keras.models import model_from_json
file = open('exported_regressor_v1.json','r')
model = file.read()
file.close()
regression_model = model_from_json(model)
regression_model.load_weights('exported_regressor_v1.h5')

single_df = predictors[1,:] # Get a station sample

import numpy as np
test_station = np.array([single_df]) # Format to np array
test_air_quality_prediction = regression_model.predict(test_station) # Runs the prediction
print(abs(test_air_quality_prediction))
################################# Result ####################################################
# --Tested with predictors[0,:] in which the expected result was 0 and we found 2.105       #
# we also tested with predictors[1,:] in which the expected result was 1 and we found 0.507 #
#############################################################################################


################################ Exporting ##########################################
reg = regressor.build_fn().to_json() # Format to json
with open('exported_regressor_v1.json','w') as json_file:
    json_file.write(reg) # Exports it
reg = regressor.build_fn().save_weights('exported_regressor_v1.h5') # Do same to the weights

################################  Loading     ########################################
from keras.models import model_from_json
file = open('exported_regressor_v1.json','r')
model = file.read()
file.close()
regression_model = model_from_json(model)
regression_model.load_weights('exported_regressor_v1.h5')
regression_model.predict(test_station[:])



