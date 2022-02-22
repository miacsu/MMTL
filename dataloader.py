from __future__ import print_function, division
from torch.utils.data import Dataset
from utils import read_csv_mt
import numpy as np
import csv
import random


class CNN_Data(Dataset):
    """
    csv files ./lookuptxt/*.csv contains MRI filenames along with demographic and diagnosis information 
    """
    def __init__(self, Data_dir, stage, dataset, cross_index, start, end, seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        self.Data_list, self.Label_list, self.demor_list = read_csv_mt('./lookupcsv/{}.csv'.format(dataset))
        train_data_list = list()
        train_label_list = list()
        train_demor_list = list()
        test_data_list = list()
        test_label_list = list()
        test_demor_list = list()
        with open("./lookupcsv/{}.csv".format(dataset), 'r') as file:
            f_reader = csv.reader(file)
            for i, row in enumerate(f_reader):
                if i == 0:
                    continue
                if end == -1 or start <= i - 1 <= end:
                    test_data_list.append(row[0])
                    test_label_list.append(0 if row[1] == "CN" else 1)
                    test_demor_list.append(int(row[2]))
                else:
                    train_data_list.append(row[0])
                    train_label_list.append(0 if row[1] == "CN" else 1)
                    train_demor_list.append(int(row[2]))
        num = end - start
        if stage == 'valid':
            if cross_index != 9:
                self.Data_list = train_data_list[num * cross_index:num * (cross_index + 1)]
                self.Label_list = train_label_list[num * cross_index:num * (cross_index + 1)]
                self.demor_list = train_demor_list[num * cross_index:num * (cross_index + 1)]
            else:
                self.Data_list = train_data_list[:num]
                self.Label_list = train_label_list[:num]
                self.demor_list = train_demor_list[:num]
        elif stage == 'train':
            if cross_index != 9:
                self.Data_list = train_data_list[:num * cross_index] + train_data_list[num * (cross_index + 1):]
                self.Label_list = train_label_list[:num * cross_index] + train_label_list[num * (cross_index + 1):]
                self.demor_list = train_demor_list[:num * cross_index] + train_demor_list[num * (cross_index + 1):]
            else:
                self.Data_list = train_data_list[num:]
                self.Label_list = train_label_list[num:]
                self.demor_list = train_demor_list[num:]
        else:
            self.Data_list = test_data_list
            self.Label_list = test_label_list
            self.demor_list = test_demor_list

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        demor = self.demor_list[idx]
        data = np.load(self.Data_dir + self.Data_list[idx] + ".npy").astype(np.float32)
        data = np.expand_dims(data, axis=0)
        return data, label, np.asarray(demor).astype(np.float32)

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.Label_list)), float(self.Label_list.count(0)), float(self.Label_list.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.Label_list]
        return weights, count0 / count1
