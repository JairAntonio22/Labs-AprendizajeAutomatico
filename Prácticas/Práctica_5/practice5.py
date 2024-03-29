import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import graphviz
from graphviz import Source
import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib.colors import ListedColormap

#Configurar figuras con directorio para guardarlas
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False,
plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)
#######################################
#Inducir arbol de decision usando dataset de:
# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
def iris_procedure():
    iris = datasets.load_iris()
    X = iris.data[:,:2]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree_clf = tree.DecisionTreeClassifier(max_depth=3)
    tree_clf.fit(X_train, y_train)

    #Dibujar modelo del árbol de decisión
    export_graphviz(
            tree_clf,
            out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
            feature_names=iris.feature_names[2:],
            class_names=iris.target_names,
            rounded=True,
            filled=True
        )

    Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))

    #Visualizar modelo al instalar el paquete graphviz,
    #y convertir archivo DOT en PNG con comando:
    # dot -Tpng iris_tree.dot -o iris_tree.png
    #Probar árbol de decisión final con conjuntos de Train y Test
    #Calcular precisión del modelo
    precision = tree_clf.score(X_test, y_test) * 100
    print("IRIS DECISION TREE PRECISION:")
    print(precision)

    plt.figure(figsize=(8, 4))
    plot_decision_boundary(tree_clf, X, y)
    plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
    plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
    plt.text(1.40, 1.0, "Depth=0", fontsize=15)
    plt.text(3.2, 1.80, "Depth=1", fontsize=13)
    save_fig("iris_decision_tree_decision_boundaries_plot")
    plt.show()

def wine_procedure():
    wine = datasets.load_wine()
    X = wine.data[:,:2]
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree_clf = tree.DecisionTreeClassifier(max_depth=3)
    tree_clf.fit(X_train, y_train)

    #Dibujar modelo del árbol de decisión
    export_graphviz(
            tree_clf,
            out_file=os.path.join(IMAGES_PATH, "wine_tree.dot"),
            feature_names=wine.feature_names[:2],
            class_names=wine.target_names,
            rounded=True,
            filled=True
        )

    Source.from_file(os.path.join(IMAGES_PATH, "wine_tree.dot"))

    #Visualizar modelo al instalar el paquete graphviz,
    #y convertir archivo DOT en PNG con comando:
    # dot -Tpng iris_tree.dot -o iris_tree.png
    #Probar árbol de decisión final con conjuntos de Train y Test
    #Calcular precisión del modelo
    precision = tree_clf.score(X_test, y_test) * 100
    print("WINE DECISION TREE PRECISION:")
    print(precision)

    plt.figure(figsize=(8, 4))
    plot_decision_boundary(tree_clf, X, y, axes=[10, 16, 0, 6], iris=False)
    save_fig("wine_decision_tree_decision_boundaries_plot")
    plt.show()

def breast_procedure():
    breast = datasets.load_breast_cancer()
    X = breast.data[:,:2]
    y = breast.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree_clf = tree.DecisionTreeClassifier(max_depth=3)
    tree_clf.fit(X_train, y_train)

    #Dibujar modelo del árbol de decisión
    export_graphviz(
            tree_clf,
            out_file=os.path.join(IMAGES_PATH, "breast_cancer_tree.dot"),
            feature_names=breast.feature_names[:2],
            class_names=breast.target_names,
            rounded=True,
            filled=True
        )

    Source.from_file(os.path.join(IMAGES_PATH, "breast_cancer_tree.dot"))

    #Visualizar modelo al instalar el paquete graphviz,
    #y convertir archivo DOT en PNG con comando:
    # dot -Tpng iris_tree.dot -o iris_tree.png
    #Probar árbol de decisión final con conjuntos de Train y Test
    #Calcular precisión del modelo
    precision = tree_clf.score(X_test, y_test) * 100
    print("BREAST CANCER DECISION TREE PRECISION:")
    print(precision)

    plt.figure(figsize=(8, 4))
    plot_decision_boundary(tree_clf, X, y, axes=[5, 30, 5, 40], iris=False)
    save_fig("breast_cancer_decision_tree_decision_boundaries_plot")
    plt.show()

##Hacer lo mismo para wine y breast_cancer
#wine: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
#breast_cancer: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

iris_procedure()
wine_procedure()
breast_procedure()
