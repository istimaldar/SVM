import os
from os.path import isfile, join

import pickle

import ovr
from kernels import GAUSSIAN

if __name__ == '__main__':
    machine = ovr.OVR()
    X = []
    t = 0
    for name in [f for f in os.listdir('train_data/') if isfile(join('train_data/', f))]:
        if t == 60:
            break
        t += 1
        with open('train_data/{}'.format(name), 'rb') as f:
            X.append(pickle.load(f))
    Y = [i for i in range(60)]
    machine.train(60, GAUSSIAN, [2], X, Y, 2)
    print(machine.classify(X[2]))
