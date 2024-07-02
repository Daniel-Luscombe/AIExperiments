import pandas as pd
import numpy as np

# Load the dataset
data_path = "/Users/DLuscombe/Library/Mobile Documents/com~apple~CloudDocs/Northeastern/Projects/AIExperiments/Titanic/Dataset/"
df = pd.read_csv(data_path + 'train.csv')

# Data Preprocessing
#remove columns that are not useful
df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId', 'Parch', 'SibSp'])
#add bias column
df.insert(loc=1, column='bias', value=1)
# Using map function convert male and female to 1, 0
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
#remove rows with missing data
df = df.dropna()

# find averages for age and fare
avg_age = df.loc[:, 'Age'].mean()
avg_fare = df.loc[:, 'Fare'].mean()

#init data to matrix and vector values
w_vector = np.array([-1, 2, 1, avg_age, avg_fare])
X_matrix = np.array(df.iloc[1:,0:6])
y_vector = df['Survived'].to_numpy()


