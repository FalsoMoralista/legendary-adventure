import pandas as pd


df = pd.read_csv('air_quality_Nov2017.csv')
df = df.drop('Latitude', axis=1)
df = df.drop('Longitude', axis=1)
df = df.drop('DateTime', axis=1)
df = df.drop('Generated', axis=1)
df = df.dropna()

#The code below will split the data between the predictor attributes and "real values" (that are expected to the model output)

predictors = df.iloc[:,2:11].values

palau_reial = df.loc[df['Station'] == 'Barcelona - Palau Reial']
palau_reial = palau_reial.iloc[:,1:11]

observ_fabra = df.loc[df['Station'] == 'Barcelona - Observ Fabra']
observ_fabra = observ_fabra.iloc[:,1:11]

eixample = df.loc[df['Station'] == 'Barcelona - Eixample']
eixample = eixample.iloc[:,1:11]

vall_hebron = df.loc[df['Station'] == 'Barcelona - Vall Hebron']
vall_hebron = vall_hebron.iloc[:,1:11]

gracia = df.loc[df['Station'] == 'Barcelona - Gràcia']
gracia = gracia.iloc[:,1:11]

palau_air = palau_reial.iloc[:,0].values
observ_air = observ_fabra.iloc[:,0].values
eixample_air = eixample.iloc[:,0].values
vall_air = vall_hebron.iloc[:,0].values
gracia_air = gracia.iloc[:,0].values

palau_reial = palau_reial.iloc[:,1:9].values
observ_fabra = observ_fabra.iloc[:,1:9].values
eixample = eixample.iloc[:,1:9].values
vall_hebron = vall_hebron.iloc[:,1:9].values
gracia = gracia.iloc[:,1:9].values

# Now we need to encode the categorical values (transform them into numbers)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder1 = LabelEncoder()

palau_air = labelEncoder1.fit_transform(palau_air)
observ_air = labelEncoder1.fit_transform(observ_air)
eixample_air = labelEncoder1.fit_transform(eixample_air)
vall_air = labelEncoder1.fit_transform(vall_air)
gracia_air = labelEncoder1.fit_transform(gracia_air)

labelEncoder2 = LabelEncoder()

predictors[:,0] = labelEncoder2.fit_transform(predictors[:,0])
predictors[:,1] = labelEncoder2.fit_transform(predictors[:,1])
predictors[:,3] = labelEncoder2.fit_transform(predictors[:,3])
predictors[:,4] = labelEncoder2.fit_transform(predictors[:,4])
predictors[:,6] = labelEncoder2.fit_transform(predictors[:,6])
predictors[:,7] = labelEncoder2.fit_transform(predictors[:,7])


palau_reial[:,0] = labelEncoder2.fit_transform(palau_reial[:,0])
palau_reial[:,1] = labelEncoder2.fit_transform(palau_reial[:,1])
palau_reial[:,3] = labelEncoder2.fit_transform(palau_reial[:,3])
palau_reial[:,4] = labelEncoder2.fit_transform(palau_reial[:,4])
palau_reial[:,6] = labelEncoder2.fit_transform(palau_reial[:,6])
palau_reial[:,7] = labelEncoder2.fit_transform(palau_reial[:,7])

observ_fabra[:,0] = labelEncoder2.fit_transform(observ_fabra[:,0])
observ_fabra[:,1] = labelEncoder2.fit_transform(observ_fabra[:,1])
observ_fabra[:,3] = labelEncoder2.fit_transform(observ_fabra[:,3])
observ_fabra[:,4] = labelEncoder2.fit_transform(observ_fabra[:,4])
observ_fabra[:,6] = labelEncoder2.fit_transform(observ_fabra[:,6])
observ_fabra[:,7] = labelEncoder2.fit_transform(observ_fabra[:,7])

eixample[:,0] = labelEncoder2.fit_transform(eixample[:,0])
eixample[:,1] = labelEncoder2.fit_transform(eixample[:,1])
eixample[:,3] = labelEncoder2.fit_transform(eixample[:,3])
eixample[:,4] = labelEncoder2.fit_transform(eixample[:,4])
eixample[:,6] = labelEncoder2.fit_transform(eixample[:,6])
eixample[:,7] = labelEncoder2.fit_transform(eixample[:,7])

vall_hebron[:,0] = labelEncoder2.fit_transform(vall_hebron[:,0])
vall_hebron[:,1] = labelEncoder2.fit_transform(vall_hebron[:,1])
vall_hebron[:,3] = labelEncoder2.fit_transform(vall_hebron[:,3])
vall_hebron[:,4] = labelEncoder2.fit_transform(vall_hebron[:,4])
vall_hebron[:,6] = labelEncoder2.fit_transform(vall_hebron[:,6])
vall_hebron[:,7] = labelEncoder2.fit_transform(vall_hebron[:,7])

gracia[:,0] = labelEncoder2.fit_transform(gracia[:,0])
gracia[:,1] = labelEncoder2.fit_transform(gracia[:,1])
gracia[:,3] = labelEncoder2.fit_transform(gracia[:,3])
gracia[:,4] = labelEncoder2.fit_transform(gracia[:,4])
gracia[:,6] = labelEncoder2.fit_transform(gracia[:,6])
gracia[:,7] = labelEncoder2.fit_transform(gracia[:,7])

hotEncoder = OneHotEncoder(categorical_features=[0,1,3,4,6,7])
palau_reial = hotEncoder.fit_transform(palau_reial).toarray()
observ_fabra = hotEncoder.fit_transform(observ_fabra).toarray()
eixample = hotEncoder.fit_transform(eixample).toarray()
vall_hebron = hotEncoder.fit_transform(vall_hebron).toarray()
gracia = hotEncoder.fit_transform(gracia).toarray()

predictors = hotEncoder.fit_transform(predictors).toarray()
#-- Now we build our network
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
import numpy as np

input_layer= Input(shape=(80,))
# Since we aren't using the sequential model, we need to inform the layer that
# cames before the layer that we are creating
hidden_layer1 = Dense(units=40,activation='relu')(input_layer)
hidden_layer2 = Dense(units=40,activation='relu')(hidden_layer1)
# Since we have five outputs, we need to create five output layers
output_layer1 = Dense(units=1,activation='linear')(hidden_layer2)
output_layer2 = Dense(units=1,activation='linear')(hidden_layer2)
output_layer3 = Dense(units=1,activation='linear')(hidden_layer2)
output_layer4 = Dense(units=1,activation='linear')(hidden_layer2)
output_layer5 = Dense(units=1,activation='linear')(hidden_layer2)

model = Model(inputs=input_layer, 
              outputs=[output_layer1, output_layer2, output_layer3, output_layer4, output_layer5])
model.compile(optimizer='adam',
              loss='mean_squared_error')

palau_reial = np.ndarray(palau_reial)

model.fit(x=[palau_reial, observ_fabra, eixample_air, vall_hebron, gracia], y=[palau_air, observ_air, eixample_air,vall_air,gracia_air], epochs=10000,batch_size=2853)

prediction_na, prediction_eu, prediction_jp = model.predict(predictors)




