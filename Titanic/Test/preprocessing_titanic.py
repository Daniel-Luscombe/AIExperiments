import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load the datasets
train_url ='https://raw.githubusercontent.com/Daniel-Luscombe/AIExperiments/main/Titanic/Test/train.csv'
test_url = 'https://raw.githubusercontent.com/Daniel-Luscombe/AIExperiments/main/Titanic/Test/test.csv'
df_train = pd.read_csv(train_url)
df_test = pd.read_csv(test_url)

#Using KDE to estimate missing age data
age = df_train['Age'].dropna()
age_data = age.to_numpy()
from scipy.stats import norm
#KDE function
def kde(x, sigma, data):
    n = len(data)
    mixture = 0
    for i in range(n):
        mixture = mixture + norm.pdf(x, data[i], sigma)
    return mixture/n
#Finding range
x_range = np.linspace(min(age_data)-10, max(age_data)+0.3, 100)
#Sigma for KDE
sig = 3
#KDE PDF
kde_pdf = kde(x_range, sig, age_data)

#Performing Rejection Sampling
from scipy.stats import uniform
from numpy.random import rand
# Interpolating KDE
def p(x):
    return np.interp(x, x_range, kde_pdf)
def q(x):
    return uniform.pdf(x, loc=min(age_data), scale=max(age_data) + 30)
#Choosing a k where k*q(x) >= p(x)
k = 4
#Finding range
x_range = np.linspace(min(age_data)-1, max(age_data)+31, 100)
#rejection sampling algorithm
def rej_smpl(n):
    samples = []
    while len(samples) < n:
        sample = np.random.uniform(min(age_data), max(age_data) + 30)
        if (k*q(sample)*rand()) < p(sample):
            samples.append(sample)
    return samples
#generating 1000 samples
final_samples = rej_smpl(1000)
sample_mean = np.mean(final_samples)

#Preprocessing
df_train_main = df_train.copy()
df_train_main = df_train_main.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
#replace age column with new data
df_train_main['Age'] = df_train_main['Age'].fillna(np.random.choice(final_samples))
print(df_train_main.shape)
#dropping NA value
df_train_main = df_train_main.dropna()
print(df_train_main.shape)
#initialising y vector
y = np.array(df_train_main['Survived']).reshape(-1,1)
#dropping Irrelivent data - Ticket & Cabin to be added later
df_train_main = df_train_main.drop(['Survived'], axis = 1)
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
Phi = np.hstack((oh_sex, oh_embarked, X_scale, bias))
print(Phi.shape)
#Splitting data for validation
#80/20 split between training and rest of data
[Phi_train, Phi_rest, y_train, y_rest] = train_test_split(Phi, y, test_size = .2)

#Preprocessing test data
PASSENGER_ID = df_test['PassengerId']
#dropping Irrelivent data - Ticket & Cabin to be added later
df_test = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
#one-hot encoding catagorical features
oh_sex_test = pd.get_dummies(df_test['Sex'], drop_first=True, dtype=int)
oh_embarked_test = pd.get_dummies(df_test['Embarked'], drop_first=True, dtype=int)
#dropping catagorical features
df_test = df_test.drop(['Sex', 'Embarked'], axis = 1)
#converting to numpy and scaling data
oh_sex_test = oh_sex_test.to_numpy()
oh_embarked_test = oh_embarked_test.to_numpy()
X_test = df_test.to_numpy()
X_scale_test = preprocessing.scale(X_test)
#Concatinating data and adding bias column
bias = np.ones((X_scale_test.shape[0], 1))
TEST_SET = np.hstack((bias, oh_sex_test, oh_embarked_test, X_scale_test))