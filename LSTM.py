import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import Metrics as m
import matplotlib.pyplot as plt
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


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    shuffle=False)


dVal = pd.read_csv('process/test.csv')
dVal['GHI'] = dVal.GHI * 60

Xval = dVal[varsRegs].values
Xval = scaler.transform(Xval)
yVal = dVal.ghi


Xval = Xval.reshape((Xval.shape[0], Xval.shape[1], 1))



# Definir función para crear y entrenar el modelo
def create_and_train_model(X_train, y_train, X_test, y_test, layers, dropout_rate, modelname, epochs=30, batch_size=80, ):
    print(f"modelo {modelname} ")
    model = Sequential()
    
    model = Sequential()
    #model.add(Dense(layers[0], input_shape=(X_train.shape[1],), activation='relu'))
    model.add(LSTM(layers[0], activation='relu', input_shape=(X_train.shape[1], 1)))
    
    for layer_size in layers[1:]:
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='relu'))
    

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_train, 
                        y_train,
                        validation_data=(X_test, y_test), 
                        epochs=epochs, 
                        verbose=1)
       
    # joblib.dump(model, f"models/{modelname}_2.joblib")
    pred  = model.predict(Xval).reshape(-1)
    rrmse = m.rrmsd(yVal, pred)
    return history, rrmse

# Configuraciones de modelos para probar
configurations = [
    {"layers": [2, 8, 16], "dropout_rate": 0.3},
    {"layers": [4, 16, 32], "dropout_rate": 0.3},
    {"layers": [8, 32, 64], "dropout_rate": 0.3},
    {"layers": [16, 8, 2], "dropout_rate": 0.3},
    #{"layers": [32, 128, 256], "dropout_rate": 0.3},
]

# Variables para almacenar los resultados
histories = []
rrmses = []

# Entrenar modelos con diferentes configuraciones
for i, config in enumerate(configurations):
    history,rrmse = create_and_train_model(X_train, 
                                     y_train, 
                                     X_test, 
                                     y_test, 
                                     config["layers"], 
                                     config["dropout_rate"], 
                                     modelname=f"model_{i}")
    histories.append(history)
    rrmses.append(rrmse)
    print(f'Configuración: {config}, RRMSE: {rrmse}')


    # m.rrmsd(yVal, prediction.reshape(-1))
    # m.rrmsd(yVal, prediction.reshape(-1))


# # Entrenar modelos con diferentes configuraciones
# for i, config in enumerate(configurations):
#     history, rrmse = create_and_train_model(X_train, y_train, X_test, y_test, config["layers"], config["dropout_rate"], modelname=f"model_{i}")
#     histories.append(history)
#     rrmses.append(rrmse)
#     print(f'Configuración: {config}, RRMSE: {rrmse}')

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


