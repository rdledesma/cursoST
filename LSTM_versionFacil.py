import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

d = pd.read_csv('process/train.csv')
# Generate sample data

d['GHI'] = d.GHI * 60
varsRegs = ['TOA', 'Clear sky GHI', 'Clear sky BHI', 'Clear sky DHI',
       'Clear sky BNI', 'GHI', 'BHI', 'DHI', 'BNI', 'Reliability', 'sza',
       'summer/winter split', 'tco3', 'tcwv', 'AOD BC', 'AOD DU', 'AOD SS',
       'AOD OR', 'AOD SU', 'AOD NI', 'AOD AM', 'AOD SO', 'Snow probability',
       'fiso', 'fvol', 'fgeo', 'albedo', 'Cloud coverage', 'Cloud type']



X = d[varsRegs]
scaler = joblib.load('models/scaler.joblib')
X = scaler.transform(X)
y = d.ghi

# Reshape input to be [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)







# Define the LSTM model
model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X, y, epochs=10, verbose=1)



dVal = pd.read_csv('process/test.csv')
dVal['GHI'] = dVal.GHI * 60

Xval = dVal[varsRegs].values
Xval = scaler.transform(Xval)
yVal = dVal.ghi




# Realiza la predicción
test_input = Xval.reshape((Xval.shape[0], Xval.shape[1], 1))
predicted_value = model.predict(test_input, verbose=1)

dVal['ghiPred'] = predicted_value



import Metrics as m

true = dVal.ghi
pred = dVal.GHI 
print(m.rrmsd(true, pred))

