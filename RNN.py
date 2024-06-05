import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import Metrics as m
import joblib
from sklearn.preprocessing import StandardScaler

dTrain = pd.read_csv('process/train.csv')
dTrain['GHI'] = dTrain.GHI * 60
dTrain['DHI'] = dTrain.DHI * 60

X = dTrain[['GHI','DHI','AOD OR','Cloud coverage']].values
y = dTrain.ghi.values

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.joblib')





# define the model architecture
model = Sequential()
model.add(Dense(2,  input_shape=(X_scaled.shape[1], ), activation='linear'))
model.add(Dense(8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))


print(model.summary())

model.compile(loss='mse', optimizer='adam')# train the model

model.fit(X_scaled, y, batch_size=100, epochs=50)

joblib.dump(model, 'models/MLP.joblib')



dTrain['pred'] = model.predict(X_scaled)


plt.plot(dTrain.ghi)
plt.plot(dTrain.GHI)
plt.plot(dTrain.pred)
plt.show()




dTest = pd.read_csv('process/test.csv')
dTest['GHI'] = dTest.GHI * 60
dTest['DHI'] = dTest.DHI * 60

X_test = dTest[['GHI','DHI']].values
y_test = dTest.ghi.values



dTest['pred'] = model.predict(X_test)



trueTrain = dTrain.ghi.values
predTrain = dTrain.pred.values

trueTest = dTest.ghi.values
predTest = dTest.pred.values
camsTest = dTest.GHI.values

print("Train")
print(f'rrmsd {m.rrmsd(trueTrain,predTrain)}')

print("Test")
print(f'rrmsd CAMS {m.rrmsd(trueTest, camsTest)}')
print(f'rrmsd Adapted {m.rrmsd(trueTest,predTest)}')


#plt.plot(dTest.ghi)A
#plt.plot(dTest.GHI)
#plt.plot(dTest.pred)
#plt.show()
