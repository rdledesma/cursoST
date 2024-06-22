import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import Metrics as m
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dTrain = pd.read_csv('process/train.csv')
dTrain['GHI'] = dTrain.GHI * 60

varsRegs = ['TOA', 'Clear sky GHI', 'Clear sky BHI', 'Clear sky DHI',
       'Clear sky BNI', 'GHI', 'BHI', 'DHI', 'BNI', 'Reliability', 'sza',
       'summer/winter split', 'tco3', 'tcwv', 'AOD BC', 'AOD DU', 'AOD SS',
       'AOD OR', 'AOD SU', 'AOD NI', 'AOD AM', 'AOD SO', 'Snow probability',
       'fiso', 'fvol', 'fgeo', 'albedo', 'Cloud coverage', 'Cloud type']






X = dTrain[varsRegs].values
y = dTrain.ghi.values

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.joblib')




X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)





# define the model architecture
model = Sequential()
model.add(Dense(2,  input_shape=(X_scaled.shape[1], ), activation='linear'))
model.add(Dense(8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))


print(model.summary())

model.compile(loss='mse', optimizer='adam')# train the model





# Ajustar el modelo y registrar la pérdida
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=50)

# Graficar la función de pérdida
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MSE)')
plt.title('Función de Pérdida durante el Entrenamiento y la Validación')
plt.legend()
plt.grid(True)
plt.show()












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
