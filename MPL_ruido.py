import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
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




import numpy as np

X = dTrain[varsRegs].values
y = dTrain.ghi.values

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.joblib')




X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)





# define the model architecture
model = Sequential()
model.add(Dense(10,  input_shape=(X_scaled.shape[1], ), activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20*2, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20*3, activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(29*2, activation='relu'))
model.add(Dense(1, activation='linear'))




model.compile(loss='mse', optimizer='adam')# train the model

# Ajustar el modelo y registrar la pérdida
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=45)





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

joblib.dump(model, 'models/MLP3.joblib')







import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# Definir función para crear y entrenar el modelo
def create_and_train_model(X_train, y_train, X_test, y_test, layers, dropout_rate, modelname, epochs=30, batch_size=80, ):
    print(f"modelo {modelname} ")
    model = Sequential()
    model.add(Dense(layers[0], input_shape=(X_train.shape[1],), activation='relu'))
    for layer_size in layers[1:]:
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer=Adam())
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)
    
    
    
    dTest = pd.read_csv('process/test.csv')


    dTest['GHI'] = dTest.GHI * 60
    varsRegs = ['TOA', 'Clear sky GHI', 'Clear sky BHI', 'Clear sky DHI',
           'Clear sky BNI', 'GHI', 'BHI', 'DHI', 'BNI', 'Reliability', 'sza',
           'summer/winter split', 'tco3', 'tcwv', 'AOD BC', 'AOD DU', 'AOD SS',
           'AOD OR', 'AOD SU', 'AOD NI', 'AOD AM', 'AOD SO', 'Snow probability',
           'fiso', 'fvol', 'fgeo', 'albedo', 'Cloud coverage', 'Cloud type']


    Xval = dTest[varsRegs].values
    yval = dTest.ghi.values
    
    Xval = scaler.transform(Xval)
    
    y_pred = model.predict(Xval)
    rrmse = np.sqrt(mean_squared_error(yval, y_pred)) / np.mean(yval) * 100
    
    joblib.dump(model, f"models/{modelname}_2.joblib")
    
    return history, rrmse

# Configuraciones de modelos para probar
configurations = [
    #{"layers": [2, 8, 16], "dropout_rate": 0.3},
    {"layers": [4, 16, 32], "dropout_rate": 0.3},
    {"layers": [8, 32, 64], "dropout_rate": 0.3},
    {"layers": [16, 64, 128], "dropout_rate": 0.3},
    #{"layers": [32, 128, 256], "dropout_rate": 0.3},
]

# Variables para almacenar los resultados
histories = []
rrmses = []

# Entrenar modelos con diferentes configuraciones
for i, config in enumerate(configurations):
    history, rrmse = create_and_train_model(X_train, y_train, X_test, y_test, config["layers"], config["dropout_rate"], modelname=f"model_{i}")
    histories.append(history)
    rrmses.append(rrmse)
    print(f'Configuración: {config}, RRMSE: {rrmse}')

# Graficar las funciones de pérdida
plt.figure(figsize=(12, 8))
for i, history in enumerate(histories):
    plt.plot(history.history['loss'], label=f'Entrenamiento {i+1} ({configurations[i]})')
    plt.plot(history.history['val_loss'], label=f'Validación {i+1} ({configurations[i]})')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MSE)')
plt.title('Función de Pérdida durante el Entrenamiento y la Validación para Diferentes Configuraciones')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar RRMSE para cada configuración
for i, rrmse in enumerate(rrmses):
    print(f'Configuración {i+1} ({configurations[i]} RRMSE = {rrmse}')










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
