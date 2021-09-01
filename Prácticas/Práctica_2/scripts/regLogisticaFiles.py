import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix

def default():
    df = pd.read_csv("Default.txt", sep="\t", header=0)

    binaryNum = {'Yes':1, 'No':0}

    df.default = [binaryNum[item] for item in df.default]

    df.student = [binaryNum[item] for item in df.student]

    x = df[['student', 'balance', 'income']]
    y = df[['default']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)

    y_pred = logistic_reg.predict(X_test)

    print("========Default.txt==============")
    print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))
    print("Classification Report:\n", classification_report(y_test,y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percentage = 100 * accuracy
    print("Accuracy Percentage: ", accuracy_percentage)

    conf_matrix = confusion_matrix(y_test,y_pred)

    conf_matrix = confusion_matrix(y_test,y_pred)

    cmn = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sb.heatmap(cmn, annot=True, fmt='.2f', xticklabels=('Predicted No', 'Predicted Yes'), yticklabels=('Actual No', 'Actual Yes'))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def genero():
    df = pd.read_csv("genero.txt",header=0)

    gender = {'Male':1, 'Female':0}

    df.Gender = [gender[item] for item in df.Gender]

    X = df[['Height','Weight']]
    y = df[['Gender']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)

    y_pred = logistic_reg.predict(X_test)

    print("========genero.txt==============")
    print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))
    print("Classification Report:\n", classification_report(y_test,y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percentage = 100 * accuracy
    print("Accuracy Percentage: ", accuracy_percentage)

    conf_matrix = confusion_matrix(y_test,y_pred)

    cmn = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sb.heatmap(cmn, annot=True, fmt='.2f', xticklabels=('Predicted Female', 'Predicted Male'), yticklabels=('Actual Female', 'Actual Male'))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

default()
genero()
