import pandas as pd 

dataset = pd.read_csv('../../data/autos/autos.csv', encoding='ISO-8859-1')
dataset = dataset.drop('dateCrawled',axis=1)
dataset = dataset.drop('dateCreated',axis=1)
dataset = dataset.drop('nrOfPictures',axis=1)
dataset = dataset.drop('postalCode',axis=1)
dataset = dataset.drop('lastSeen',axis=1)


# Drops attributes that will not be used
dataset['name'].value_counts()
dataset = dataset.drop('name',axis=1)
dataset['seller'].value_counts()
dataset = dataset.drop('seller',axis=1)
dataset['offerType'].value_counts()
dataset = dataset.drop('offerType',axis=1)

dataset = dataset[dataset.price > 10] # Remove inconsistent values
dataset = dataset[dataset.price < 350000] # Remove inconsistent values

# Localizates the biggest occurency of empty values for each attribute below
dataset.loc[pd.isnull(dataset['vehicleType'])]
dataset['vehicleType'].value_counts() # limousine

dataset.loc[pd.isnull(dataset['gearbox'])]
dataset['gearbox'].value_counts() # manuell

dataset.loc[pd.isnull(dataset['model'])]
dataset['model'].value_counts() # golf

dataset.loc[pd.isnull(dataset['fuelType'])]
dataset['fuelType'].value_counts() # benzin

dataset.loc[pd.isnull(dataset['notRepairedDamage'])]
dataset['notRepairedDamage'].value_counts() # nein
###############################################################################
# This is the values that we will used to fill the empty boxes in our dataset
values = {'vehicleType':'limousine',
          'gearbox':'manuell', 
          'model':'golf', 
          'fuelType':'benzin', 
          'notRepairedDamage':'nein'}

dataset = dataset.fillna(value=values)
###############################################################################
previsors = dataset.iloc[:,1:13].values
real_price = dataset.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsors = LabelEncoder()

previsors[:,0] = labelencoder_previsors.fit_transform(previsors[:,0])
previsors[:,1] = labelencoder_previsors.fit_transform(previsors[:,1])
previsors[:,3] = labelencoder_previsors.fit_transform(previsors[:,3])
previsors[:,5] = labelencoder_previsors.fit_transform(previsors[:,5])
previsors[:,8] = labelencoder_previsors.fit_transform(previsors[:,8])
previsors[:,9] = labelencoder_previsors.fit_transform(previsors[:,9])
previsors[:,10] = labelencoder_previsors.fit_transform(previsors[:,10])


hotencoder = OneHotEncoder(categorical_features=[0,1,3,5,8,9,10])
previsors = hotencoder.fit_transform(previsors).toarray()

# Building the neural network to predict prices

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

def build():
    regressor = Sequential()
    regressor.add(Dense(units=158, activation='relu',input_dim=316))
    regressor.add(Dense(units=158, activation='relu'))
    regressor.add(Dense(units=1,activation='linear'))
    regressor.compile(loss='mean_absolute_error', 
                      optimizer='adam',
                      metrics = ['mean_absolute_error'])
    return regressor

regressor = KerasRegressor(build_fn=build,
                           epochs=100,
                           batch_size=300)
results= cross_val_score(estimator=regressor,
                         X = previsors,
                         y = real_price,
                         cv=2, scoring = 'neg_median_absolute_error')



mean=results.mean()
sd = results.std()