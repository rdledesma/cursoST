#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:58:27 2024

@author: dario
"""

import pandas as pd
import joblib
import Metrics as m

import numpy as np
model = joblib.load('models/model_1.joblib')


dTest = pd.read_csv('process/test.csv')


dTest['GHI'] = dTest.GHI * 60
varsRegs = ['TOA', 'Clear sky GHI', 'Clear sky BHI', 'Clear sky DHI',
       'Clear sky BNI', 'GHI', 'BHI', 'DHI', 'BNI', 'Reliability', 'sza',
       'summer/winter split', 'tco3', 'tcwv', 'AOD BC', 'AOD DU', 'AOD SS',
       'AOD OR', 'AOD SU', 'AOD NI', 'AOD AM', 'AOD SO', 'Snow probability',
       'fiso', 'fvol', 'fgeo', 'albedo', 'Cloud coverage', 'Cloud type']




X = dTest[varsRegs].values
y = dTest.ghi.values

scaler = joblib.load('models/scaler.joblib')

X_test = scaler.transform(X)


y_test = dTest.ghi.values







dTest['pred'] = model.predict(X_test)


trueTest = dTest.ghi.values
predTest = dTest.pred.values
camsTest = dTest.GHI.values


print("Test")
print(f'rrmsd CAMS {m.rrmsd(trueTest, camsTest)}')
print(f'rrmsd Adapted {m.rrmsd(trueTest,predTest)}')

import matplotlib.pyplot as plt
plt.plot(dTest.ghi)
plt.plot(dTest.GHI)
plt.plot(dTest.pred)
plt.show()

