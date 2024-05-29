import pandas as pd
import matplotlib.pyplot as plt
d = pd.read_csv('cams/out.csv', header=68, sep=";")
e = pd.read_csv('measured/sla2023.csv')
f = pd.read_csv('measured/sla2024.csv')

f['TIMESTAMP'] = pd.to_datetime(f.TIMESTAMP)
f = f[f.TIMESTAMP.dt.year == 2024]

e['TIMESTAMP'] = pd.to_datetime(e.TIMESTAMP)
e = e[e.TIMESTAMP.dt.year == 2023]



s = pd.concat([e,f])

s = s[s.Acepted]

s = s[['TIMESTAMP','ghi']]
s.columns = ['date','ghi']


s = s.reset_index()


plt.plot(s.ghi)
plt.show()



d = pd.read_csv('cams/out.csv', sep=";", header=68)
d['date'] = pd.date_range(start="2023/01/01 00:00",
                          end="2024/05/21 23:59", freq="1min")
d['date'] = pd.date_range(start="2023/01/01 00:00",
                          end="2024/05/21 23:59", freq="1min")

d.drop('# Observation period', axis=1, inplace=True)
d.drop('alpha', axis=1, inplace=True)
d.drop('Cloud optical depth', axis=1, inplace=True)

s['date'] = pd.to_datetime(s.date)

d = (d.set_index('date')
      .reindex(s.date)
      .rename_axis(['date'])
      #.fillna(0)
      .reset_index())


d['ghi'] = s.ghi


import Geo
g = Geo.Geo(d.date, 
            -24.72888, 
            -65.40979, 
            gmt =0, 
            alt = 1234, 
            beta=0).df

plt.plot(g.SZA, d.ghi, '.', ms=0.5)
plt.show()



d['sza'] = g.SZA.values
d.to_csv('process/data.csv', index=False)

d[d.date.dt.year == 2023].to_csv('process/train.csv', index=False)
d[d.date.dt.year == 2024].to_csv('process/test.csv', index=False)

