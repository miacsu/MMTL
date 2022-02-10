import os
import sys

from utils import read_json
from model_wrapper import CNN_Wrapper
import torch
import multiprocessing
import time
torch.backends.benchmark = True


def cnn_main(fold_index, gpu_index, process):

    if not os.path.exists("checkpoint_dir/cnn_exp{}".format(fold_index)):
        os.mkdir("checkpoint_dir/cnn_exp{}".format(fold_index))
    cnn_setting = config['cnn']

    # CNN_Wrapper is used to wrap the model and its training and testing. The function used for cross-validation is
    # CNN_Wrapper.cross_validation.

    with torch.cuda.device(gpu_index):
        cnn = CNN_Wrapper(fil_num=cnn_setting['fil_num'],
                          drop_rate=cnn_setting['drop_rate'],
                          batch_size=cnn_setting['batch_size'],
                          balanced=cnn_setting['balanced'],
                          data_dir=cnn_setting['Data_dir'],
                          learn_rate=cnn_setting['learning_rate'],
                          train_epoch=cnn_setting['train_epochs'],
                          dataset=config["dataset"],
                          external_dataset=config["external_dataset"],
                          seed=config["seed"],
                          model_name='cnn',
                          metric='accuracy')
        cnn.cross_validation(fold_index, process)


if __name__ == "__main__":
    print("Hello World!")

    process = ""
    if len(sys.argv) > 1:
        process = sys.argv[1]

    if not os.path.exists("checkpoint_dir"):
        os.mkdir("checkpoint_dir")

    # Read related parameters
    config = read_json('./config.json')
    dataset = config["dataset"]
    folds = config["folds"]
    gpus = config["gpus"]

    # to perform CNN training and testing
    for i in range(len(folds)):
        p = multiprocessing.Process(target=cnn_main, args=(folds[i], gpus[i], process, ))
        p.start()
        time.sleep(60)

    print("Main process end.")
