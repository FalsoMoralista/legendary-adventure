import pandas as pd

from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
import numpy as np

df = pd.read_csv('game-sales.csv')

df = df.drop('Other_Sales', axis = 1)
df = df.drop('Global_Sales', axis = 1)
df = df.drop('Developer',axis = 1)
df = df.drop('Name', axis = 1)
df = df.dropna(axis = 0)

df = df.loc[df['NA_Sales'] > 1]
df = df.loc[df['EU_Sales'] > 1]

predictors = df.iloc[:, [0,1,2,3,7,8,9,10,11]].values

sales_na = df.iloc[:,4].values
sales_eu = df.iloc[:,5].values
sales_jp = df.iloc[:,6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
predictors[:,0] = labelEncoder.fit_transform(predictors[:,0])
predictors[:,2] = labelEncoder.fit_transform(predictors[:,2])
predictors[:,3] = labelEncoder.fit_transform(predictors[:,3])
predictors[:,8] = labelEncoder.fit_transform(predictors[:,8])

oneHotEncoder = OneHotEncoder(categorical_features=[0,2,3,8])
predictors = oneHotEncoder.fit_transform(predictors).toarray()

###############################################################################
# In this example we will learn how to use a different model, (alternative to
# the sequential).

input_layer= Input(shape=(61,))
# Since we aren't using the sequential model, we need to inform the layer that
# cames before the layer that we are creating
hidden_layer1 = Dense(units=32,activation='sigmoid')(input_layer)
hidden_layer2 = Dense(units=32,activation='sigmoid')(hidden_layer1)
# Since we have three outputs, we need to create three output layers
output_layer1 = Dense(units=1,activation='linear')(hidden_layer2)
output_layer2 = Dense(units=1,activation='linear')(hidden_layer2)
output_layer3 = Dense(units=1,activation='linear')(hidden_layer2)

model = Model(inputs=input_layer, 
              outputs=[output_layer1,output_layer2,output_layer3])
model.compile(optimizer='adam',
              loss='mean_squared_error')
model.fit(predictors, [sales_eu, sales_jp, sales_na], epochs=10000,batch_size=258)

prediction_na, prediction_eu, prediction_jp = model.predict(predictors)
###############################################################################
# This will be used later to fill null values with a probabilistic distribution
def fillna():    
    arr = df['Publisher'].value_counts()
    index =  arr2.index
    values = arr2.values
    
    probabilities = []

    total = 0;
    for i in values:
    total += i

    j = 0
    while j < len(values):
    probabilities.append(values[j]/total)
    j+= 1
    
    np.random.choice(a=index, p=probabilities)

    columns = df.columns.tolist()
    for c in columns:
        c = str(c)
        print(df[c].value_counts())    

