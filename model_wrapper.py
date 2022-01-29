from torch.utils.data import Dataset, DataLoader
from model import _CNN
from dataloader import CNN_Data
from loss import ConRegGroupLoss
from utils import matrix_sum, get_acc, get_MCC, get_confusion_matrix, write_raw_score, squared_error
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim


class CNN_Wrapper:
    def __init__(self,
                 fil_num,
                 drop_rate,
                 seed,
                 batch_size,
                 balanced,
                 data_dir,
                 learn_rate,
                 train_epoch,
                 dataset,
                 external_dataset,
                 model_name,
                 metric):

        """
            :param fil_num:    channel number
            :param drop_rate:  dropout rate
            :param seed:       random seed
            :param batch_size: batch size for training CNN
            :param balanced:   balanced could take value 0 or 1, corresponding to different approaches to handle data
                               imbalance, see self.prepare_dataloader for more details
            :param model_name: give a name to the model
            :param metric:     metric used for saving model during training, can be either 'accuracy' or 'MCC' for
                               example, if metric == 'accuracy', then the time point where validation set has best
                               accuracy will be saved
        """

        self.epoch = 0
        self.seed = seed
        self.Data_dir = data_dir
        self.learn_rate = learn_rate
        self.train_epoch = train_epoch
        self.balanced = balanced
        self.batch_size = batch_size
        self.dataset = dataset
        self.external_dataset = external_dataset
        self.model_name = model_name
        self.cross_index = None
        self.get_con_reg_group_loss = ConRegGroupLoss()
        self.eval_metric = get_acc if metric == 'accuracy' else get_MCC
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, self.cross_index)
        self.model = _CNN(fil_num=fil_num, drop_rate=drop_rate).cuda()
        self.optimal_epoch = self.epoch
        self.optimal_valid_mse = 99999.0
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0.0
        self.frequency_dict = None

    def cross_validation(self, cross_index):
        self.cross_index = cross_index
        with open("lookupcsv/{}.csv".format(self.dataset), 'r') as csv_file:
            num = len(list(csv.reader(csv_file))) // 10
        start = int(self.cross_index * num)
        end = start + (num - 1)
        with open(self.checkpoint_dir + 'valid_result.txt', 'w') as file:
            file.write('')
        self.prepare_dataloader(start, end)
        self.train()
        self.test()

    def prepare_dataloader(self, start, end):
        train_data = CNN_Data(self.Data_dir, stage='train', dataset=self.dataset, cross_index=self.cross_index,
                              start=start, end=end, seed=self.seed)
        valid_data = CNN_Data(self.Data_dir, stage='valid', dataset=self.dataset, cross_index=self.cross_index,
                              start=start, end=end, seed=self.seed)
        test_data = CNN_Data(self.Data_dir, stage='test', dataset=self.dataset, cross_index=self.cross_index,
                             start=start, end=end, seed=self.seed)
        self.frequency_dict = self.get_con_reg_group_loss.update(train_data.Label_list, train_data.demor_list)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        # the following if else blocks represent two ways of handling class imbalance issue
        if self.balanced == 1:
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif self.balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                               num_workers=0)
        self.valid_dataloader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def train(self):
        # Train the model
        print("Fold {} is training ...".format(self.cross_index))

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))
        self.criterion_clf = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).cuda()
        self.criterion_reg = nn.SmoothL1Loss(reduction='mean').cuda()
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch = 0

        while self.epoch < self.train_epoch:
            self.train_model_epoch()
            valid_matrix, valid_mse = self.valid_model_epoch()
            with open(self.checkpoint_dir + "valid_result.txt", 'a') as file:
                file.write(str(self.epoch) + ' ' + str(valid_matrix) + " " +
                           str(round(self.eval_metric(valid_matrix), 4)) + ' ' + str(valid_mse) + ' ' + '\n')
            print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix)
            print('eval_metric:', "%.4f" % self.eval_metric(valid_matrix), 'and mean squared error ', valid_mse)
            self.save_checkpoint(valid_matrix, valid_mse)
            self.epoch += 1
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric,
              self.optimal_valid_matrix)
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, labels, demors in self.train_dataloader:
            inputs, labels, demors = inputs.cuda(), labels.cuda(), demors.cuda()
            self.model.zero_grad()
            loss = torch.tensor(0.0, requires_grad=True).cuda()
            clf_output, reg_output, per_loss = self.model(inputs)
            clf_loss = self.criterion_clf(clf_output, labels)
            reg_loss = self.criterion_reg(reg_output, torch.unsqueeze(demors, dim=1))
            con_reg_group_loss = self.get_con_reg_group_loss.apply(reg_output, demors, self.frequency_dict, labels)
            loss = loss + clf_loss + reg_loss + torch.mean(con_reg_group_loss) + torch.mean(per_loss)
            loss.backward()
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            valid_matrix = [[0, 0], [0, 0]]
            mse = 0.0
            for inputs, labels, demors in self.valid_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                clf_output, reg_output, _ = self.model(inputs)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(clf_output, labels))
                mse += squared_error(reg_output, demors)
            mse /= (valid_matrix[0][0] + valid_matrix[0][1] + valid_matrix[1][0] + valid_matrix[1][1])
        return valid_matrix, mse

    def save_checkpoint(self, valid_matrix, valid_mse):
        # Choose the optimal model. The performance of AD detection task is prioritized
        if (self.eval_metric(valid_matrix) > self.optimal_valid_metric) or \
                (self.eval_metric(valid_matrix) == self.optimal_valid_metric and valid_mse < self.optimal_valid_mse):
            self.optimal_epoch = self.epoch
            self.optimal_valid_matrix = valid_matrix
            self.optimal_valid_metric = self.eval_metric(valid_matrix)
            self.optimal_valid_mse = valid_mse
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith("pth"):
                        os.remove(os.path.join(self.checkpoint_dir, File))
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name,
                                                                     self.optimal_epoch))

    def test(self):
        # Test the model
        print('Fold {} is testing ... '.format(self.cross_index))

        # Load the optimal model
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name,
                                                                   self.optimal_epoch)))
        self.model.train(False)

        with torch.no_grad():
            for dataset in self.dataset + self.external_dataset:
                for stage in ['train', 'valid', 'test']:
                    data_dir = self.Data_dir
                    if dataset != self.dataset:
                        if stage != 'test':
                            continue
                        data_dir = data_dir.replace('ADNI1', dataset)
                        data = CNN_Data(data_dir, stage=stage, dataset=dataset, cross_index=self.cross_index, start=0,
                                        end=-1, seed=self.seed)
                        dataloader = DataLoader(data, batch_size=2, shuffle=False)
                    elif stage == 'train':
                        dataloader = self.train_dataloader
                    elif stage == 'valid':
                        dataloader = self.valid_dataloader
                    else:
                        dataloader = self.test_dataloader
                    f_clf = open(self.checkpoint_dir + 'raw_score_clf_{}_{}.txt'.format(dataset, stage), 'w')
                    f_reg = open(self.checkpoint_dir + 'raw_score_reg_{}_{}.txt'.format(dataset, stage), 'w')
                    matrix = [[0, 0], [0, 0]]
                    mse = 0.0
                    for idx, (inputs, labels, demors) in enumerate(dataloader):
                        inputs, labels = inputs.cuda(), labels.cuda()
                        clf_output, reg_output, _ = self.model(inputs)
                        write_raw_score(f_clf, clf_output, labels)
                        matrix = matrix_sum(matrix, get_confusion_matrix(clf_output, labels))
                        mse += squared_error(reg_output, demors)
                        write_raw_score(f_reg, reg_output, demors)
                    mse /= (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
                    print(dataset + "-" + stage + ' confusion matrix ', matrix)
                    print('accuracy:', "%.4f" % self.eval_metric(matrix), 'and mean squared error ', mse)
                    f_clf.close()
                    f_reg.close()
