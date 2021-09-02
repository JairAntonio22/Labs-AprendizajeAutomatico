import pandas as pd
import numpy as np

def grad_desc():


df = pd.read_csv("Default.txt", sep="\t", header=0)

binaryNum = {'Yes':1, 'No':0}

df.default = [binaryNum[item] for item in df.default]

df.student = [binaryNum[item] for item in df.student]

x = df[['student', 'balance', 'income']]
y = df[['default']]
xs = np.array([row[:] for row in x])
ys = np.array([row[:] for row in y])

print("========Default.txt==============")
print('Confusion Matrix: \n')
print('Classification Report: \n')
print('Accuracy Percentage: \n')
print('intercept value:')
print('coeffiecients:')

df = pd.read_csv("genero.txt", header=0)

gender = {'Male':1, 'Female':0}

df.Gender = [gender[item] for item in df.Gender]

X = df[['Height','Weight']]
y = df[['Gender']]
xs = np.array([row[:] for row in x])
ys = np.array([row[:] for row in y])

print("========genero.txt==============")
print('Confusion Matrix: \n')
print('Classification Report: \n')
print('Accuracy Percentage: \n')
print('intercept value:')
print('coeffiecients:')
