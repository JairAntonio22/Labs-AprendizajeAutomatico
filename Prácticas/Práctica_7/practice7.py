''' Practica no. 7 '''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve


def main():
    ''' Main script '''
    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.80, random_state=42
    )

    names = [
        'svm linear', 'svm poly', 'svm rbf', 'svm sigmoid',
        'log reg', 'knn', 'bayes'
    ]

    clfs = []
    clfs.append(SVC(kernel='linear'))
    clfs.append(SVC(kernel='poly', degree=2, coef0=1))
    clfs.append(SVC(kernel='rbf', gamma=1))
    clfs.append(SVC(kernel='sigmoid', coef0=1))

    clfs.append(LogisticRegression(max_iter=1000, multi_class='ovr'))
    clfs.append(KNeighborsClassifier(n_neighbors=3))
    clfs.append(GaussianNB())

    fig_count = 1

    for name, clf in zip(names, clfs):
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test) * 100
        y_pred = clf.predict(X_test)
        cmat = confusion_matrix(y_test, y_pred)

        print('===== %s =====' % name.center(15, ' '))
        print('precision: %5.2f%%\n' % score)
        print(cmat)
        print()

        # Falta calcular los fpr y tpr para las graficas ROC
        '''
        fpr, tpr = roc_curve(y, ...

        plt.figure(fig_count)
        plt.scatter([fpr, 0, 0, 1, 1], [tpr, 0, 1, 0, 1])
        plt.plot([0, fpr, 1], [0, tpr, 1])
        plt.savefig('ROC%s.png' % name)
        fig_count += 1
        '''


if __name__ == '__main__':
    main()
