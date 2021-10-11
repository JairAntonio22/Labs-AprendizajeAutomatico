''' Practica no. 7 '''

from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc


def main():
    ''' Main script '''
    digits = load_digits()
    X = digits.data
    y = digits.target

    y = label_binarize(y, classes=[i for i in range(10)])
    n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.80, random_state=42
    )

    names = [
        'svm linear', 'svm poly', 'svm rbf', 'svm sigmoid',
        'log reg', 'knn', 'bayes'
    ]

    clfs = [
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(SVC(kernel='poly', degree=2, coef0=1)),
        OneVsRestClassifier(SVC(kernel='rbf', gamma=10)),
        OneVsRestClassifier(SVC(kernel='sigmoid', coef0=-1)),
        OneVsRestClassifier(LogisticRegression(max_iter=1000, multi_class='ovr')),
        OneVsRestClassifier(KNeighborsClassifier(n_neighbors=1)),
        OneVsRestClassifier(GaussianNB())
    ]

    fig_count = 1
    colors = [
        'orangered', 'darkorange', 'gold', 'yellowgreen', 'darkcyan',
        'steelblue', 'slategrey', 'mediumslateblue', 'mediumorchid', 'deeppink'
    ]

    for name, clf in zip(names, clfs):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) * 100

        print('===== %s =====' % name.center(15, ' '))
        print('precision: %5.2f%%\n' % score)

        if name in ['knn', 'bayes']:
            y_score = clf.predict_proba(X_test)

        else:
            y_score = clf.fit(X_train, y_train).decision_function(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='black', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='black', linestyle=':', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve %s' % name)
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('ROC%s.png' % name)

        fig_count += 1


if __name__ == '__main__':
    main()
