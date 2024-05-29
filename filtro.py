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
from datetime import timedelta
%matplotlib qt

d = pd.read_csv('measured/BDD_sla_2023.csv')
d['TIMESTAMP'] = pd.to_datetime(d.TIMESTAMP)


df = d[['TIMESTAMP', 'PSP_Avg']].copy()
df['ghi'] = df.PSP_Avg/1000/7.50e-6
df.columns = ['TIMESTAMP', 'PSP_Avg', 'ghi']

df['TIMESTAMP'] = df['TIMESTAMP'] + timedelta(hours=3)

dfGeo = Geo.Geo(df.TIMESTAMP, 
                lat= -24.72888, #-24.72888
                long= -65.40979,#  -65.40979
                gmt = 0,
                alt = 1234  , beta = 0).df # 3348  #1190

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


df = df[df.CTZ > 0.17]

plt.plot(df.TIMESTAMP, df.ghi, '--')
plt.plot(df.TIMESTAMP[df.Acepted == True], df.ghi[df.Acepted == True], '.-')

plt.plot(df.CTZ[df.Acepted == True], df.ghi[df.Acepted == True], '.', ms=0.5)

plt.plot(df.CTZ[df.Acepted == True], df.ghi[df.Acepted == True]/df.TOA[df.Acepted == True], '.', ms=0.5)



#df.to_csv('BDDsla2023_NQC.csv', index=False)

#%% EJEMPLO DE SKLEARN DE SENSio