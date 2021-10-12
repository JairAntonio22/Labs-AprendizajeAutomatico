''' Script para encontrar el mejor valor para k con el dataset digits '''

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


def main():
    ''' Main script '''
    digits = load_digits()
    X = digits.data
    y = digits.target

    y = label_binarize(y, classes=[i for i in range(10)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.80, random_state=42
    )

    neighbors = [k for k in range(1, 359)]
    scores = []

    for k in range(1, 359):
        clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=k))
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test) * 100)

    print('(maxscore, k) =', max(zip(scores, neighbors)))

    plt.plot(neighbors, scores)
    plt.savefig('bestK.png')
    plt.show()


if __name__ == '__main__':
    main()
