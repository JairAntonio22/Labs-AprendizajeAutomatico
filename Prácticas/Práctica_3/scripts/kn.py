import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def credit():
    dataset = pd.read_csv("Default.txt", sep='\t',)
    yes_no = {'Yes': 1, 'No': 0}
    dataset.default = [yes_no[item] for item in dataset.default]
    dataset.student = [yes_no[item] for item in dataset.student]

    X = np.array(dataset[['student', 'balance', 'income']])
    y = np.array(dataset[['default']])

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

    neighbors = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]
    scores = []

    for value in neighbors:
        neigh = KNeighborsClassifier(algorithm='brute', n_neighbors=value)
        neigh.fit(X_train, y_train.ravel())

        score = neigh.score(X_test, y_test)
        y_pred = neigh.predict(X_test)

        print(f'=== neighbors value {value} ===')
        print(f'precision: ', score)
        print(f'confusion matrix:\n', confusion_matrix(y_test, y_pred))

        scores.append(score)

    plt.figure(1)
    plt.scatter(neighbors, scores)
    plt.plot(neighbors, scores)
    plt.show()

def scatterGenero(figurenum, xs, ys):
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

def genero():
    dataset = pd.read_csv("genero.txt")
    gender = {'Male': 1, 'Female': 0}
    dataset.Gender = [gender[item] for item in dataset.Gender]

    X = np.array(dataset[['Height','Weight']])
    y = np.array(dataset[['Gender']])

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

    neighbors = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]
    scores = []

    for value in neighbors:
        neigh = KNeighborsClassifier(algorithm='brute', n_neighbors=value)
        neigh.fit(X_train, y_train.ravel())

        score = neigh.score(X_test, y_test)
        y_pred = neigh.predict(X_test)

        print(f'=== neighbors value {value} ===')
        print(f'precision: ', score)
        print(f'confusion matrix:\n', confusion_matrix(y_test, y_pred))

        scores.append(score)

    plt.figure(1)
    plt.scatter(neighbors, scores)
    plt.plot(neighbors, scores)

    scatterGenero(2, X_train, y_train)

    k = (20 + 50) // 2
    neigh = KNeighborsClassifier(algorithm='brute', n_neighbors=k)
    neigh.fit(X_train, y_train.ravel())
    y_pred = neigh.predict(X_test)

    scatterGenero(3, X_test, y_pred)
    plt.show()

credit()
genero()
