import os
import random
from os.path import isfile, join

import pickle

import ovr
from kernels import GAUSSIAN

if __name__ == '__main__':
    wrong = 72
    while (wrong / 72) > 0.05:
        machine = ovr.OVR()
        X = []
        Y = []
        t = 0
        for name in [f for f in os.listdir('train_data/') if isfile(join('train_data/', f))]:
            if t == 55:
                break
            t += 1
            with open('train_data/{}'.format(name), 'rb') as f:
                X.append(pickle.load(f))
            with open('diagnoses/{}'.format(name), 'r') as f:
                data = f.readline().rstrip()
                Y.append(int(data))
        sigma = 2
        C = 100
        machine.train(40, GAUSSIAN, [sigma], X, Y, C)
        wrong = 0
        X_n = []
        Y_n = []
        for name in [f for f in os.listdir('train_data/') if isfile(join('train_data/', f))]:
            with open('train_data/{}'.format(name), 'rb') as f:
                X_n.append(pickle.load(f))
            with open('diagnoses/{}'.format(name), 'r') as f:
                data = f.readline().rstrip()
                Y_n.append(int(data))
        for x, y in zip(X_n, Y_n):
            if machine.classify(x) != y:
                wrong += 1
        print(wrong / 72, sigma, C)
