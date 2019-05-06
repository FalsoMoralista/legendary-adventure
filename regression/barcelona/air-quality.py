import pandas as pd

data = pd.read_csv('../../data/barcelona-data-sets/air_quality_Nov2017.csv')
data['Station'].value_counts()

