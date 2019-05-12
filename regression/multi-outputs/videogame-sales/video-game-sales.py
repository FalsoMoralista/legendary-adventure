import pandas as pd

from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
import numpy as np

df = pd.read_csv('game-sales.csv')

df = df.drop('Other_Sales', axis = 1)
df = df.drop('Global_Sales', axis = 1)
df = df.drop('Developer',axis = 1)
df = df.dropna(axis = 0)



arr = df['Publisher'].tolist()

arr2 = df['Publisher'].value_counts()
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

nan = df.isnull().any()


matrix = df.iloc[:,:].values

shape = np.shape(matrix)

for columns in matrix[0,:]:
    print(columns)