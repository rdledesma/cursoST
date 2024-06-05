#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:58:27 2024

@author: dario
"""

import pandas as pd
import joblib
import Metrics as m

model = joblib.load('models/MLP.joblib')


dTest = pd.read_csv('process/test.csv')


dTest['GHI'] = dTest.GHI * 60
dTest['DHI'] = dTest.DHI * 60
X = dTest[['GHI','DHI','AOD OR','Cloud coverage']].values

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
