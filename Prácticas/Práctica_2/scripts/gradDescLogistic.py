import numpy as np
import pandas as pd
from math import e

def sigmoid(t):
    return 1 / (1 + e**(-t))

# Evaluación del modelo dadas las entradas xs y coeficientes bs
def h(xs, bs, intercept=0):
    return sigmoid(np.dot(xs, bs) + intercept)

# Cálculo de gradiente, esto se transcribió directamente desde la fórmula
def grad_mse(xs, ys, bs):
    return np.dot(h(xs, bs) - ys, xs)

# Implementación de descenso de gradiente
def grad_desc(xs, ys, alpha):
    xs = np.insert(xs, 0, 1, axis=1) # Valor para la intersección
    gs = bs = np.ones(xs.shape[1])
    steps = 0

    while np.linalg.norm(gs) > 0.01 and steps < 10_000:
        gs = grad_mse(xs, ys, bs)
        print(gs)
        bs -= alpha * gs
        steps += 1

    return bs[0], bs[1:] # intersección, coeficientes

#Gráfica para Género
def scatterGenero(figurenum,xs, ys):
    x_male = []
    y_male = []
    x_female = []
    y_female = []

    for x, y in zip(xs, ys):
        if y == 1:
            x_male.append(x[0])
            y_male.append(x[1])
        else:
            x_female.append(x[0])
            y_female.append(x[1])

    plt.figure(figurenum)
    plt.scatter(x_male, y_male, color='blue')
    plt.scatter(x_female, y_female, color='red')

# Script principal
if __name__ == '__main__':

    df = pd.read_csv("Default.txt", sep="\t", header=0)

    binaryNum = {'Yes':1, 'No':0}

    df.default = [binaryNum[item] for item in df.default]

    df.student = [binaryNum[item] for item in df.student]

    X = np.array(df[['student', 'balance', 'income']])
    y = np.array(df[['default']])

    intercept, bs = grad_desc(X, y, 0.00001)

    print("========Default.txt==============")
    print('Confusion Matrix: \n')
    print('Classification Report: \n')
    print('Accuracy Percentage: \n')
    print('intercept value:', intercept)
    print('coeffiecients:', bs)


    df = pd.read_csv("genero.txt", header=0)

    gender = {'Male':1, 'Female':0}

    df.Gender = [gender[item] for item in df.Gender]

    X = np.array(df[['Height','Weight']])
    y = np.array(df[['Gender']])

    intercept, bs = grad_desc(X, y, 0.00001)

    print('=== genero.txt ===')
    print('Confusion Matrix: \n')
    print('Classification Report: \n')
    print('Accuracy Percentage: \n')
    print('intercept value:', intercept)
    print('coeffiecients:', bs)










def genero():
    df = pd.read_csv("genero.txt", header=0)

    gender = {'Male':1, 'Female':0}

    df.Gender = [gender[item] for item in df.Gender]

    X = np.array(df[['Height','Weight']])
    y = np.array(df[['Gender']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    print("========genero.txt==============")

    # Training the model
    b0, b1 = grad_desc_Genero(X_train, y_train)

    # Making predictions
    X_test_norm = normalizar(X_test)
    y_pred = predict(X_test_norm, b0, b1)
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

    scatterGenero(1,X_train,y_train)

    scatterGenero(2,X_test, y_pred)
    plt.show()

    accuracy = 0

    for i in range(len(y_pred)):
        if y_pred[i] == y_test.iloc[i]:
            accuracy += 1

    print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))
    print('Classification Report: \n', classification_report(y_test,y_pred))
    print(f"Accuracy = {accuracy / len(y_pred)}")
