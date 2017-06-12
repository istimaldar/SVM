import os
import re
import urllib.request

from bs4 import BeautifulSoup
import requests
import wfdb
import biosppy


def check_point_for_extremum_in_neighborhood(data, element_num, neighborhood_size):
    neighborhood = data[element_num - neighborhood_size if element_num - neighborhood_size > 0 else 0: element_num] + \
                   data[element_num + 1: element_num + neighborhood_size if element_num + neighborhood_size < len(data)
                   else len(data) - 1]
    minimum = True
    maximum = True
    for i in neighborhood:
        if data[element_num] <= i:
            maximum = False
        if data[element_num] >= i:
            minimum = False
    return minimum or maximum


def find_extremums(data):
    result = []
    for n, element in enumerate(data):
        if n == 0 or n == len(data) - 1 or (data[n - 1] < element > data[n + 1]) or\
                (data[n - 1] > element < data[n + 1]):
            if check_point_for_extremum_in_neighborhood(data, n, 20):
                result.append((n, element))
    return result


PREFIX = 'https://physionet.org/physiobank/database/twadb/'
# NAMES = ("001a", "001b", "001c", "001d")


def load(NAMES):
    for NAME in NAMES:
        urllib.request.urlretrieve("{url}{name}.hea".format(url=PREFIX, name=NAME),
                                   "{}.hea".format(NAME))
        record = wfdb.rdsamp("data\{}".format(NAME), sampto=10000)
        wfdb.plotrec(record)
        signals = [list(i) for i in zip(*record.p_signals)]
        extremimus = [find_extremums(element) for element in signals]
        print(extremimus)
        os.remove("{}.dat".format(NAME))
        os.remove("{}.hea".format(NAME))


def get_data_source():
    soup = BeautifulSoup(requests.get(PREFIX).text)
    return list(sorted(set(re.findall(r'a href=[]', str(soup)))))


if __name__ == '__main__':
    load(['twa01'])
