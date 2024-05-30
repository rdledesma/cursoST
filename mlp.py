# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:26:44 2024

@author: Cony
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import Metrics as m
%matplotlib qt

dTrain = pd.read_csv('process/train.csv')
dTest = pd.read_csv('process/test.csv')

date_Train = dTrain['date']
date_Test = dTest['date']

dTrain.drop(['date'], axis=1, inplace=True)
dTest.drop(['date'], axis=1, inplace=True)

cols = ['TOA', 'Clear sky GHI', 'Clear sky BHI', 'Clear sky DHI',
       'Clear sky BNI', 'GHI', 'BHI', 'DHI', 'BNI', 'GHI no corr', 'BHI no corr', 'DHI no corr', 'BNI no corr']

for c in cols:
    dTrain[c] *= 60
    dTest[c] *= 60

# Crear dos instancias de MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Ajustar y transformar las características y la variable objetivo por separado
X_train = dTrain.drop(columns=['ghi'])  # reemplaza 'target_column' con el nombre de tu columna objetivo
y_train = dTrain['ghi'].values.reshape(-1, 1)  # reemplaza 'target_column' con el nombre de tu columna objetivo

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Hacer lo mismo para los datos de prueba
X_test = dTest.drop(columns=['ghi'])  # reemplaza 'target_column' con el nombre de tu columna objetivo
y_test = dTest['ghi'].values.reshape(-1, 1)  # reemplaza 'target_column' con el nombre de tu columna objetivo

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

reg = LinearRegression().fit(X_train_scaled, y_train_scaled)
pred_train_scaled = reg.predict(X_train_scaled) 
pred_test_scaled = reg.predict(X_test_scaled)

# Aplicar la transformación inversa utilizando los objetos scaler_X y scaler_y
pred_train = scaler_y.inverse_transform(pred_train_scaled)
pred_test = scaler_y.inverse_transform(pred_test_scaled)



plt.plot(pred_train)
plt.plot(y_train)

plt.plot(pred_test)
plt.plot(y_test)



print("Train")
print(f'rrmsd {m.rrmsd(y_train,pred_train)}')

print("Test")
print(f'rrmsd {m.rrmsd(y_test, pred_test)}')

