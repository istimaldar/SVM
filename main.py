import os
import random
from os.path import isfile, join

import pickle

import ovr
from kernels import GAUSSIAN

if __name__ == '__main__':
    machine = ovr.OVR()
    X = []
    Y = []
    t = 0
    for name in [f for f in os.listdir('train_data/') if isfile(join('train_data/', f))]:
        if t == 55:
            break
        t += 1
        with open('diagnoses/{}'.format(name), 'w') as f:
            f.write(str(int(random.random() * 40)))
        with open('train_data/{}'.format(name), 'rb') as f:
            X.append(pickle.load(f))
        with open('diagnoses/{}'.format(name), 'r') as f:
            data = f.readline().rstrip()
            Y.append(int(data))
    machine.train(55, GAUSSIAN, [2], X, Y, 4)
    wrong = 0
    for x in range(55):
        if machine.classify(X[x]) != Y[x]:
            wrong += 1
    print(wrong)
