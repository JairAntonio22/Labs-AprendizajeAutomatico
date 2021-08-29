import numpy as np
import pandas as pd
import seaborn as seabornInstance
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def genero():
    #Cargar el dataset de genero.txt
    dataset = pd.read_csv('genero.txt')

    #Dividir los datos en X y Y
    X = dataset['Height'].values.reshape(-1,1)
    y = dataset['Weight'].values.reshape(-1,1)

    #Dividir 80% de los datos en el Training Set y 20% en el Test Set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #Comenzar a entrenar el algoritmo con el LinearRegression y el fit()
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    #El modelo encuentra el mejor valor para el intercept y el slope para crear la línea
    #Para el intercept
    print('=== genero.txt ===')
    print('Intercept value: ', regressor.intercept_)
    #Para el slope:
    print('Coefficients: ', regressor.coef_)

    #Ya que se entrenó, se realizan predicciones
    y_pred = regressor.predict(X_test)

    #Se compara el valor actual para X_Test con los predicted values
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    print(df)

    #Se evalúa el desempeño del algoritmo. 
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

def carros():
    #Cargar el dataset de genero.txt
    dataset = pd.read_csv('mtcars.txt', sep=' ')

    #Dividir los datos en X, Y, y Z
    x = dataset['disp'].values.reshape(-1,1)
    y = dataset['wt'].values.reshape(-1,1)
    z = dataset['hp'].values.reshape(-1,1)

    X = [[a[0],b[0]] for a, b in zip(x,y)]

    #Dividir 80% de los datos en el Training Set y 20% en el Test Set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #Comenzar a entrenar el algoritmo con el LinearRegression y el fit()
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    #El modelo encuentra el mejor valor para el intercept y el slope para crear la línea
    #Para el intercept
    print('=== mtcars.txt ===')
    print('Intercept value: ', regressor.intercept_)
    #Para el slope:
    print('Coefficients: ', regressor.coef_)

    #Ya que se entrenó, se realizan predicciones
    y_pred = regressor.predict(X_test)

    #Se compara el valor actual para X_Test con los predicted values
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    print(df)

    #Se evalúa el desempeño del algoritmo. 
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

genero()
carros()
