import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

# Load the datasets
df_train = pd.read_csv(os.getcwd() + '/Titanic/Test/train.csv')
df_test = pd.read_csv(os.getcwd() + '/Titanic/Test/test.csv')

#Preprocessing
#dropping NaN features
df_train = df_train.dropna()
#initialising y vector
y = np.array(df_train['Survived']).reshape(-1,1)
#dropping Irrelivent data - Ticket & Cabin to be added later
df_train_main = df_train.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
#one-hot encoding catagorical features
oh_sex = pd.get_dummies(df_train_main['Sex'], drop_first=True, dtype=int)
oh_embarked = pd.get_dummies(df_train_main['Embarked'], drop_first=True, dtype=int)
#dropping catagorical features
df_train_main = df_train_main.drop(['Sex', 'Embarked'], axis = 1)
#converting to numpy and scaling data
oh_sex = oh_sex.to_numpy()
oh_embarked = oh_embarked.to_numpy()
X = df_train_main.to_numpy()
X_scale = preprocessing.scale(X)
#Concatinating data and adding bias column
bias = np.ones((X_scale.shape[0], 1))
Phi = np.hstack((bias, oh_sex, oh_embarked, X_scale))

#Splitting data for validation
#80/20 split between training and rest of data
[Phi_train, Phi_rest, y_train, y_rest] = train_test_split(Phi, y, test_size = .2)
#50/50 split between validation and test data
[Phi_val, Phi_test, y_val, y_test] = train_test_split(Phi_rest, y_rest, test_size = .5)