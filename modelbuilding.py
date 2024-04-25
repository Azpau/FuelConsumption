import pandas as pd
import numpy as np

rawData = pd.read_csv('FuelConsumption (1).csv')
rawData=rawData.drop(['Year'],axis=1)
rawData=rawData.drop(['Unnamed: 10','Unnamed: 11','Unnamed: 12'],axis=1)
rawData['ENGINE SIZE']=rawData['ENGINE SIZE'].astype('category')
rawData['CYLINDERS']=rawData['CYLINDERS'].astype('category')


# Import label encoder
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
rawData['MAKE']= label_encoder.fit_transform(rawData['MAKE'])
rawData['MODEL']= label_encoder.fit_transform(rawData['MODEL'])
rawData['VEHICLE CLASS']= label_encoder.fit_transform(rawData['VEHICLE CLASS'])
rawData['TRANSMISSION']= label_encoder.fit_transform(rawData['TRANSMISSION'])
rawData['FUEL']= label_encoder.fit_transform(rawData['FUEL'])


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
rawData[['COEMISSIONS_Scaled']] = scaler.fit_transform(rawData[['COEMISSIONS ']])
# Print the scaled DataFrame
rawData=rawData.drop(['COEMISSIONS '],axis=1)
print(rawData)

from sklearn.model_selection import train_test_split
X=rawData.drop('FUEL CONSUMPTION', axis=1)
y=rawData['FUEL CONSUMPTION']
print(X.head())
print(y.head())
#training and testing split using all feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2) #stratify only for classification not regression
from sklearn.linear_model import LinearRegression
modellr = LinearRegression()
modellr.fit(X_train, y_train)
import pickle
pickle.dump(modellr,open("FuelConsumptionModel.h5","wb"))
print("Model Saved")
