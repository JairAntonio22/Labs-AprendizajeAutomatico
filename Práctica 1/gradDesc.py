import numpy as np
import pandas as pd

def h(xs, bs):
    return np.dot(xs, bs)

def grad_mse(xs, ys, bs):
    return (2 / len(bs)) * np.dot(np.matrix.transpose(xs), h(xs, bs) - ys)

def grad_desc(xs, ys):
    alpha = 0.5
    bs = np.ones(xs.shape[1])
    gs = bs
    steps = 0

    while np.linalg.norm(gs) > 0.1 and steps < 10:
        print(np.linalg.norm(gs))
        gs = grad_mse(xs, ys, bs)
        bs -= alpha * gs
        steps += 1

    return bs

if __name__ == '__main__':
    dataset = np.loadtxt("genero.txt", skiprows=1, usecols=(1,2), delimiter=',')
    
    xs = np.array([row[:-1] for row in dataset])
    ys = np.array([row[-1] for row in dataset])

    print('result:', grad_desc(xs, ys))
