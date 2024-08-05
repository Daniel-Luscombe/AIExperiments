import pandas as pd
import numpy as np
import Perceptron as p

# Load the dataset
data_path = "/Users/DLuscombe/Library/Mobile Documents/com~apple~CloudDocs/Northeastern/Projects/AIExperiments/Titanic/Dataset/"
df = pd.read_csv(data_path + 'train.csv')
df_test = pd.read_csv(data_path + 'test.csv')

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
y_vector = df['Survived'].to_numpy()
X_matrix = np.array(df.drop(columns='Survived').iloc[:,0:6])

#run perceptron
alpha = 0.08
max_iter = 10000000
w = p.start_perceptron(X_matrix, y_vector, w_vector, alpha, max_iter)

# Perceptron prediction function
def perceptron_predict(X, w):
    return np.where(np.dot(X, w) >= 0.0, 1, 0)

# Calculate accuracy
def calculate_accuracy(X, y, w):
    predictions = perceptron_predict(X, w)
    accuracy = np.mean(predictions == y)
    return accuracy

# Test the accuracy
accuracy = calculate_accuracy(X_matrix, y_vector, w)
print("Accuracy:", accuracy)

# Data Preprocessing for test data
final_submission = df_test['PassengerId']
#remove columns that are not useful
df_test = df_test.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId', 'Parch', 'SibSp'])
#add bias column
df_test.insert(loc=1, column='bias', value=1)
# Using map
df_test['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

print(df.head)
print(df_test.head)