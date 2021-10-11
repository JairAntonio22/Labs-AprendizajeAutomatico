''' Script para encontrar el mejor valor para k con el dataset digits '''

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    ''' Main script '''
    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.80, random_state=42
    )

    neighbors = [k for k in range(1, 350)]
    scores = []

    for k in range(1, 350):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test) * 100)

    print('(maxscore, k) =', max(zip(scores, neighbors)))

    plt.plot(neighbors, scores)
    plt.savefig('bestK.png')
    plt.show()


if __name__ == '__main__':
    main()
