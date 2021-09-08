import numpy as np

# Evaluación del modelo dadas las entradas xs y coeficientes bs
def h(xs, bs):
    return np.dot(xs, bs)

# Cálculo de gradiente, esto se transcribió directamente desde la fórmula
def grad_mse(xs, ys, bs, n):
    res = h(xs, bs)
    res = res - ys
    res = np.dot(res, xs)
    res = (2 / n) * res
    return res
    # return (2 / n) * np.dot(h(xs, bs) - ys, xs)

# Implementación de descenso de gradiente
def grad_desc(xs, ys, alpha):
    xs = np.insert(xs, 0, 1, axis=1) # Valor para la intersección 
    gs = bs = np.ones(xs.shape[1])
    steps = 0

    while np.linalg.norm(gs) > 0.01 and steps < 1:
        gs = grad_mse(xs, ys, bs, xs.shape[0])
        bs -= alpha * gs
        steps += 1

    return bs[0], bs[1:] # intersección, coeficientes

# Script principal
if __name__ == '__main__':
    data = np.loadtxt("genero.txt", skiprows=1, usecols=(1,2), delimiter=',')
    xs = np.array([row[:-1] for row in data])
    ys = np.array([row[-1] for row in data])
    print(ys)

    intercept, bs = grad_desc(xs, ys, 0.00001)

    print('=== genero.txt ===')
    print('intercept value:', intercept)
    print('coeffiecients:', bs)

    data = np.loadtxt("mtcars.txt", skiprows=1, usecols=(4,5,7))
    xs = np.array([[row[0], row[2]] for row in data])
    ys = np.array([row[1] for row in data])

    intercept, bs = grad_desc(xs, ys, 0.0000001)

    print('=== mtcars.txt ===')
    print('intercept value:', intercept)
    print('coeffiecients:', bs)
