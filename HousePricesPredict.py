import csv
import numpy as np
import pandas as pd
import sympy
import random
from fancyimpute import (
    BiScaler,
    KNN,
    NuclearNormMinimization,
    SoftImpute,
    SimpleFill
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model

#Hyper-Parameters
k = 3
n_components = 45
hidden_layers = 50

path = r"C:\Users\marce\Documents\GitHub\DataScience\data\\"
file = "train.csv"
startRow = 2;

data = pd.read_csv(path+file);

n = data.shape[0]
m = data.shape[1]

target = data.iloc[:,m-1]
data = data.iloc[:,0:m-2]
m = data.shape[1]

categoricalIdx = []
data2 = data.copy()
for i in range(0, m):
    if (isinstance(data.iloc[:,i][0],str) or np.isnan(data.iloc[:,i][0])):
        categoricalIdx.append(i);
        modifiedoCol = pd.factorize(data.iloc[:,i],na_sentinel=-2)
        data2.iloc[:,i] = modifiedoCol[0]+1;

data2 = data2.replace(-1, np.nan)

#for i in range(0, m):
#    for j in range(0, n):
#        if (isinstance(data.iloc[:,i][0],str)):
#            print("i=",i," j=",j)

knnImpute = KNN(k)
data_knnImp = knnImpute.complete(data2.values)
data_knnImp =  StandardScaler().fit_transform(data_knnImp)


pca = PCA(n_components)
principalComponents = pca.fit_transform(data_knnImp)

pd.DataFrame(principalComponents)

train = principalComponents[0:1000,:]
test = target[0:1000]
XTest = principalComponents[1000:n,:]
YTest = target[1000:n]
pd.DataFrame(train)
#nn = MLPRegressor(hidden_layers,  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
#    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=10000, shuffle=True,
#    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#nfit = nn.fit(train, test)

#reg = linear_model.Ridge (alpha = .5)
reg = linear_model.Lasso(alpha = 0.5)
reg.fit(train, test)

test_y = reg.predict(XTest)
print(sum((YTest[:]-test_y[:])**2))

path2 = r"C:\Users\marce\Documents\GitHub\DataScience\data\\"
file2 = "test.csv"
dataSub = pd.read_csv(path2+file2);

n2 = dataSub.shape[0]
m2 = dataSub.shape[1]

data2sub = dataSub.copy()

for i in range(0, m2):
    if (isinstance(dataSub.iloc[:,i][0],str) or np.isnan(dataSub.iloc[:,i][0])):
        categoricalIdx.append(i);
        modifiedoCol = pd.factorize(dataSub.iloc[:,i],na_sentinel=-2)
        data2sub.iloc[:,i] = modifiedoCol[0]+1;


data2sub = data2sub.replace(-1, np.nan)

knnImpute = KNN(k)
datasub_knnImp = knnImpute.complete(data2sub.values)
datasub_knnImp =  StandardScaler().fit_transform(datasub_knnImp)

pca = PCA(n_components)
principalComponentsSub = pca.fit_transform(datasub_knnImp)
pd.DataFrame(principalComponentsSub)
SubY = reg.predict(principalComponentsSub)
Submit = {'Id': range(1461,2920), 'SalePrice': SubY}
SubmitData = pd.DataFrame(Submit)
SubmitData.to_csv("Predictions.csv", encoding='utf-8', index=False)
