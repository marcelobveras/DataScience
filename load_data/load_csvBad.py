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

#Hyper-Parameters
k = 10
n_components = 20
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
nn = MLPRegressor(hidden_layers,  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=10000, shuffle=True,
    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nfit = nn.fit(train, test)

test_y = nn.predict(XTest)
sum((YTest[:]-test_y[:])**2
