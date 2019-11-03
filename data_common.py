import sys
import numpy as np
import pandas as pd
import data_metrics as metr

from keras.utils import to_categorical


# Read file
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


# Get the data
def get_data(max_len, ratio):
    posi_file = sys.argv[1]
    nega_file = sys.argv[2]

    posis = read_file(posi_file)
    negas = read_file(nega_file)

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

    print('posis num:', len(posis))
    print('negas num:', len(negas))
    # Sequence
    X = posis + negas
    # Label
    y = np.array([1] * len(posis) + [0] * len(negas), dtype=int)
    y = to_categorical(y, num_classes=2)
    X = padding(X, max_len)
    X = one_hot(X)
    return X, y


# Padding the sequence to max length
def padding(data, maxlen):
    for i in range(len(data)):
        if len(data[i]) < maxlen:
            newItem = '0' * (maxlen - len(data[i])) + data[i]
            data[i] = newItem
        if len(data[i]) > maxlen:
            newItem = data[i][:maxlen]
            data[i] = newItem
    return data


# One hot
def one_hot(data):
    ltrdict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], '0': [0, 0, 0, 0]}
    data_return = list()
    for seq in data:
        data_return.append(np.array([ltrdict[x] for x in seq]))
    return np.array(data_return)


# Save the results
def save_results(file_name):
    metr.get_results()
    auc, acc, f1, aupr, spec, precision, recall = metr.get_results()
    date_frame = pd.DataFrame(
        {'auc': auc, 'acc': acc, 'f1': f1, 'aupr': aupr, 'spec': spec, 'precision': precision, 'recall': recall})
    date_frame.to_csv('./result/' + file_name + '.csv', index=True,
                      columns=['auc', 'acc', 'f1', 'aupr', 'spec', 'precision', 'recall'],
                      float_format='%.4f')
