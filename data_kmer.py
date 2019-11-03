# -*- coding: utf-8 -*-
import sys
import numpy as np
from numpy import array
from itertools import combinations_with_replacement, permutations

ratio = 3
file_name = 'dro'+str(ratio)


# Read the file
def read_file(filename):
    dataset = []
    f = open(filename, 'r')
    while True:
        s = f.readline()
        if not s:
            break
        else:
            if '>' not in s:
                seq = s.split('\n')[0]
                dataset.append(seq)
    return dataset


def GetKmerDict(alphabet, k):
    kmerlst = []
    partkmers = list(combinations_with_replacement(alphabet, k))
    for element in partkmers:
        elelst = set(permutations(element, k))
        strlst = [''.join(ele) for ele in elelst]
        kmerlst += strlst
    kmerlst = np.sort(kmerlst)
    kmerdict = {kmerlst[i]: i for i in range(len(kmerlst))}
    return kmerdict


def GetSpectrumProfile(instances, alphabet, k):
    kmerdict = GetKmerDict(alphabet, k)
    X = []
    for sequence in instances:
        vector = GetSpectrumProfileVector(sequence, kmerdict, k)
        X.append(vector)
    X = array(X)
    return X


def GetSpectrumProfileVector(sequence, kmerdict, k):
    vector = np.zeros((1, len(kmerdict)))
    n = len(sequence)
    for i in range(n - k + 1):
        subsequence = sequence[i:i + k]
        position = kmerdict.get(subsequence)
        vector[0, position] += 1
    return list(vector[0])


def get_data_kmer(ratio):
    posis_file = sys.argv[1]
    negas_file = sys.argv[2]

    posis = read_file(posis_file)
    negas = read_file(negas_file)

    if ratio == 1:
        if len(posis) < len(negas):
            negas = negas[:len(posis)]
        else:
            posis = posis[:len(negas)]
    elif ratio == 2:
        if len(posis) < len(negas):
            negas = negas[:2 * len(posis)]
        else:
            posis = posis[:2 * len(negas)]
    elif ratio == 3:
        posis = posis
        negas = negas

    instances = array(posis + negas)
    alphabet = ['A', 'C', 'G', 'T']

    # Spectrum Profile for k=1,2,3,4,5
    for k in range(1, 2):
        X = GetSpectrumProfile(instances, alphabet, k)
        X = np.array(X)
        X = (np.linalg.inv(np.diag(np.array(np.mat(X).sum(1)).flatten()))) * np.mat(X)
        np.savetxt('./kmer/' + file_name + '.txt', X, fmt='%f')


if __name__ == '__main__':
    get_data_kmer(ratio)
