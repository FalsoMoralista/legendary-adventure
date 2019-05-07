import pandas as pd
import os
print(os.getcwd())

df = pd.read_csv('../../data/barcelona-data-sets/air_quality_Nov2017.csv')

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
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score


# Builds the feed forward neural network 
def build():
    regressor = Sequential()
    regressor.add(Dense(units=403, activation='relu', input_dim=806)) # Input/first layer
    regressor.add(Dense(units=403, activation='relu')) # Second layer
    regressor.add(Dense(units=1,activation='linear')) # Output
    regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics = ['mean_absolute_error'])
    return regressor


regressor = KerasRegressor(build_fn=build, epochs=100, batch_size=3)
results= cross_val_score(estimator=regressor, X = predictors, y = real_air_quality, cv=2, scoring = 'neg_median_absolute_error')

mean=results.mean()
sd = results.std()
print(abs(mean))

################################  Exporting     ######################################
reg = regressor.build_fn().to_json() # Format to json
with open('exported_regressor.json','w') as json_file:
    json_file.write(reg) # Exports it
reg = regressor.build_fn().save_weights('exported_regressor.h5') # Do same to the weights






