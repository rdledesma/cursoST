import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import Metrics as m

dTrain = pd.read_csv('process/train.csv')
dTrain['GHI'] = dTrain.GHI * 60
X = dTrain.GHI.values
y = dTrain.ghi.values

reg = LinearRegression().fit(X.reshape(-1,1), y)
dTrain['pred'] = reg.predict(X.reshape(-1,1))

plt.plot(dTrain.ghi)
plt.plot(dTrain.GHI)
plt.plot(dTrain.pred)
plt.show()



dTest = pd.read_csv('process/test.csv')
dTest['GHI'] = dTest.GHI * 60
dTest['pred'] = reg.predict(dTest.GHI.values.reshape(-1,1))

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




