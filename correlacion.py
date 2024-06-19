# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:09:50 2024

@author: Cony
"""
import pandas as pd
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import LSTM, Dense
#import Metrics as m
#import joblib
#from sklearn.preprocessing import StandardScaler

dTrain = pd.read_csv('process/train.csv')
dTest = pd.read_csv('process/test.csv')

cols = ['TOA', 'Clear sky GHI', 'Clear sky BHI', 'Clear sky DHI',
       'Clear sky BNI', 'GHI', 'BHI', 'DHI', 'BNI', 'GHI no corr', 
       'BHI no corr', 'DHI no corr', 'BNI no corr']

for c in cols:
    dTrain[c] *= 60
    dTest[c] *= 60

dd = dTrain.copy()
correlation_matrix = dd.corr()
# Filtra las correlaciones con la variable objetivo (GHI)

ghi_correlations = abs(correlation_matrix['ghi'])
# Ordena las correlaciones de mayor a menor

sorted_correlations = ghi_correlations.sort_values(ascending=False)
# Imprime las variables con mayor correlación con la GHI

print(round(sorted_correlations, 8))

# Crea el gráfico de calor

import seaborn as sns
plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.subplots_adjust(bottom=0.18)
plt.show()