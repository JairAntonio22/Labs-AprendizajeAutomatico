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
    print("Accuracy Percentage: ", accuracy_percentage, "%")

    conf_matrix = confusion_matrix(y_test,y_pred)

    fpr = conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[1][1])
    tpr = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])

    plt.figure(2)
    plt.scatter([fpr, 0, 0, 1, 1], [tpr, 0, 1, 0, 1])
    plt.plot([0, fpr, 1], [0, tpr, 1])
    plt.savefig('rocCreditRegLog.png')

    cmn = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sb.heatmap(cmn, annot=True, fmt='.2f', xticklabels=('Predicted No', 'Predicted Yes'), yticklabels=('Actual No', 'Actual Yes'))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusionDefaultMatRegLog.png')
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
    df = pd.read_csv("genero.txt",header=0)

    gender = {'Male':1, 'Female':0}

    df.Gender = [gender[item] for item in df.Gender]

    X = np.array(df[['Height','Weight']])
    y = np.array(df[['Gender']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)

    y_pred = logistic_reg.predict(X_test)

    print("========genero.txt==============")
    print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))
    print("Classification Report:\n", classification_report(y_test,y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percentage = 100 * accuracy
    print("Accuracy Percentage: ", accuracy_percentage, "%")

    conf_matrix = confusion_matrix(y_test,y_pred)

    fpr = conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[1][1])
    tpr = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])

    plt.figure(2)
    plt.scatter([fpr, 0, 0, 1, 1], [tpr, 0, 1, 0, 1])
    plt.plot([0, fpr, 1], [0, tpr, 1])
    plt.savefig('rocGeneroRegLog.png')

    cmn = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sb.heatmap(cmn, annot=True, fmt='.2f', xticklabels=('Predicted Female', 'Predicted Male'), yticklabels=('Actual Female', 'Actual Male'))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusionGeneroMatRegLog.png')

    scatterGenero(4, X_test, y_pred)
    plt.savefig('weightHeightRegLog.png')

    plt.show()

default()
genero()
