import os
import numpy as np


def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'electricity':
        data_path = os.path.join('../data/electricity.txt')
        fin = open(data_path)
        data = np.loadtxt(fin, delimiter=',')
    elif dataset == 'solar':
        data_path = os.path.join('../data/solar_AL.txt')
        fin = open(data_path)
        data = np.loadtxt(fin, delimiter=',')
    elif dataset == 'exchange_rate':
        data_path = os.path.join('../data/exchange_rate.txt')
        fin = open(data_path)
        data = np.loadtxt(fin, delimiter=',')
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
