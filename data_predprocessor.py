import os
from os.path import isfile, join

import biosppy
import pickle

import numpy
import wfdb
import peakutils
from matplotlib import pyplot
import itertools
import utility


def load():
    for name in [f for f in os.listdir('data/') if isfile(join('data/', f)) and '.dat' in f]:
        record = wfdb.rdsamp("data/{}".format(name[:-4]))
        wfdb.plotrec(record)
        predprocessed_data = [list(signal) for signal in zip(*record.p_signals)]
        data = [tuple(biosppy.ecg.ecg(signal=f, sampling_rate=record.fs, show=False)) for f in predprocessed_data]
        print(data)


def process_data():
    for name in [f for f in os.listdir('data/') if isfile(join('data/', f)) and '.dat' in f]:
        record = wfdb.rdsamp("data/{}".format(name[:-4]))
        processed_data = [list(signal) for signal in zip(*record.p_signals)]
        data = [tuple(biosppy.ecg.ecg(signal=f, sampling_rate=record.fs, show=False)) for f in processed_data]
        data = [{'ts': list(element[0]), 'filtered': list(element[1]), 'rpeaks': list(element[2]),
                 'templates': list(element[4]), 'heart_rate': list(element[6])} for element in data]
        with open('processed_data/{}'.format(name[:-4]), 'wb') as f:
            pickle.dump(data, f)
        print(name)


def prepare_data_for():
    for name in [f for f in os.listdir('processed_data/') if isfile(join('processed_data/', f))]:
        data = None
        with open('processed_data/{}'.format(name), 'rb') as f:
            data = pickle.load(f)
        y = [sum(i) / len(data[0]['templates']) for i in zip(*data[0]['templates'])]
        for j in range(1, len(data)):
            y = [y[n] + (sum(i) / len(data[0]['templates'])) for n, i in enumerate(zip(*data[j]['templates']))]
        y = [i / len(data) for i in y]
        temp = [abs(i) for i in y]
        temp2 = list(sorted(temp))[-5:]
        peaks = [(y[temp.index(i)], temp.index(i)) for i in temp2]
        peaks = list(itertools.chain(*peaks))
        r_distance = [j-i for i, j in zip(data[0]['rpeaks'][:-1], data[0]['rpeaks'][1:])]
        for k in range(1, len(data)):
            r_distance = [(r_distance[n] if len(r_distance) > n else 0) + j - i for n, (i, j) in
                          enumerate(zip(data[k]['rpeaks'][:-1], data[k]['rpeaks'][1:]))]
        r_distance = [i / len(data) for i in r_distance]
        distance_ev, distance_dispersion = utility.get_ev_and_dispersion(r_distance)
        r_peaks = []
        for k in range(len(data)):
            r_peak = [data[k]['ts'][i] for i in data[k]['rpeaks']]
            peaks_ev, peaks_dispersion = utility.get_ev_and_dispersion(r_peak)
            r_peaks += [peaks_ev, peaks_dispersion]
        heart_rate = data[0]['heart_rate']
        rate_ev, distance_dispersion = utility.get_ev_and_dispersion(heart_rate)
        result = peaks + [distance_ev, distance_dispersion] + r_peaks + [rate_ev, distance_dispersion]
        with open('train_data/{}'.format(name), 'wb') as f:
            pickle.dump(result, f)
        print(name)

if __name__ == '__main__':
    prepare_data_for()
