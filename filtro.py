# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Geo
import NollasQC
from scipy.optimize import curve_fit
%matplotlib qt

d = pd.read_csv('measured/BDDsla_2024.csv')
d['TIMESTAMP'] = pd.to_datetime(d.TIMESTAMP)


df = d[['TIMESTAMP', 'ghiPSP']].copy()
#df['ghi'] = df.PSP_Avg/1000/7.50e-6
df.columns = ['TIMESTAMP', 'ghi']

dfGeo = Geo.Geo(df.TIMESTAMP, 
                lat= -24.72, 
                long= -65.41,
                gmt = -3,
                alt = 1234  , beta = 0).df

df['CTZ'] = dfGeo.CTZ
df['SZA'] = dfGeo.SZA
df['TZ'] = dfGeo.TZ
df['TOA'] = dfGeo.TOA
df['kt'] = df.ghi/df.TOA

NollasQC.QC(df)

df.loc[df['TIMESTAMP'].dt.date == pd.to_datetime('2023-06-27').date(), 'Acepted'] = False
df.loc[
    (df['TIMESTAMP'].dt.date == pd.to_datetime('2023-05-15').date()) &
    (df['TIMESTAMP'].dt.time >= pd.to_datetime('10:48:00').time()) &
    (df['TIMESTAMP'].dt.time <= pd.to_datetime('10:51:00').time()),
    'Acepted'
] = False


plt.plot(df.TIMESTAMP, df.ghi, '--')
plt.plot(df.TIMESTAMP[df.Acepted == True], df.ghi[df.Acepted == True], '.')

df.to_csv('BDDsla2024_NQC.csv', index=False)

#%% EJEMPLO DE SKLEARN DE SENSio