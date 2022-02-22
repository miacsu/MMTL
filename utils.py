import numpy as np
from numpy import random
import json
import csv
import random
import os
import time


def load_txt(txt_dir, txt_name):
    List = []
    with open(txt_dir + txt_name, 'r') as f:
        for line in f:
            List.append(line.strip('\n').replace('.nii', '.npy'))
    return List


def get_confusion_matrix(preds, labels):
    labels = labels.data.cpu().numpy()
    preds = preds.data.cpu().numpy()
    matrix = [[0, 0], [0, 0]]
    for index, pred in enumerate(preds):
        if np.amax(pred) == pred[0]:
            if labels[index] == 0:
                matrix[0][0] += 1
            if labels[index] == 1:
                matrix[0][1] += 1
        elif np.amax(pred) == pred[1]:
            if labels[index] == 0:
                matrix[1][0] += 1
            if labels[index] == 1:
                matrix[1][1] += 1
    return matrix


def matrix_sum(A, B): 
    return [[A[0][0]+B[0][0], A[0][1]+B[0][1]],
            [A[1][0]+B[1][0], A[1][1]+B[1][1]]]


def get_acc(matrix):
    return float(matrix[0][0] + matrix[1][1]) / float(sum(matrix[0]) + sum(matrix[1]))


def get_MCC(matrix):
    TP, TN, FP, FN = float(matrix[0][0]), float(matrix[1][1]), float(matrix[0][1]), float(matrix[1][0])
    upper = TP * TN - FP * FN
    lower = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    return upper / (lower**0.5 + 0.000000001)


def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config


def write_raw_score(f, preds, labels):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')


def write_raw_score_sk(f, preds, labels):
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')


def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames = [a[0] for a in your_list[1:]]
    labels = [0 if a[1]=='CN' else 1 for a in your_list[1:]]
    return filenames, labels


def read_csv_mt(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames, labels, demors = [], [], []
    for line in your_list[1:]:
        filenames.append(line[0])
        label = 0 if line[1] == 'CN' else 1
        labels.append(label)
        demor = line[2]
        demors.append(demor)
    return filenames, labels, demors


def data_split(repe_time, dataset):
    if dataset == "ADNI1":
        with open('./lookupcsv/ADNI1.csv', 'r') as f:
            reader = csv.reader(f)
            your_list = list(reader)
    elif dataset == "ADNI2":
        with open("./lookupcsv/ADNI2.csv", 'r') as csv_file:
            f_reader = csv.reader(csv_file)
            your_list = list(f_reader)
    labels = your_list[0]
    del your_list[0]
    train = list()
    valid = list()
    test = list()
    index = [i for i in range(len(your_list))]
    random.shuffle(index)
    if dataset == 'ADNI1':
        for i in range(len(your_list) - 1):
            if index[i] <= int(len(your_list) * 0.6):
                train.append(your_list[index[i]])
            elif index[i] <= int(len(your_list) * 0.8):
                valid.append(your_list[index[i]])
            else:
                test.append(your_list[index[i]])
    for i in range(repe_time):
        folder = 'lookupcsv/exp{}/'.format(i)
        if not os.path.exists(folder):
            os.mkdir(folder)
        with open(folder + 'train.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + train)
        with open(folder + 'valid.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + valid)
        with open(folder + 'test.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + test)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def squared_error(y_hat, y_true):
    mse = 0.0
    y_hat = y_hat.data.cpu().numpy()
    for i in range(y_hat.shape[0]):
        mse += (y_true[i] - y_hat[i][0])**2
    return mse