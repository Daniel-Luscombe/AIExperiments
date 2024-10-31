import pandas as pd
import numpy as np
import Perceptron as p
from Test.preprocessing_titanic import Phi_train, y_train, Phi_test, y_test, Phi_val, y_val
import os

#run perceptron
alpha = 0.01
max_iter = 2000
w = p.start_perceptron(Phi_train, y_train, np.ones(Phi_train.shape[1]), alpha, max_iter)

# Perceptron prediction function
def perceptron_predict(X, w):
    return np.where(np.dot(X, w) >= 0.0, 1, 0)

# Calculate accuracy
def calculate_accuracy(X, y, w):
    predictions = perceptron_predict(X, w)
    accuracy = np.mean(predictions == y)
    return accuracy

# Test the accuracy
accuracy = calculate_accuracy(Phi_test, y_test, w)
print("Accuracy:", accuracy)

#Data Preprocessing for test data
df_test = pd.read_csv(os.path.join('Test/test.csv'))
final_submission = df_test['PassengerId']
final_submission = pd.concat([final_submission, pd.DataFrame(perceptron_predict(Phi_val, w))], axis=1)
final_submission.columns = ['PassengerId', 'Survived']
final_submission.to_csv(os.path.join('Submission/perceptron_prediction.csv'), index=False)