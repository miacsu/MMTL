import os
import sys

from utils import read_json
from model_wrapper import CNN_Wrapper
import torch
import multiprocessing
import time
torch.backends.benchmark = True


def cnn_main(process, gpu_index, fold_index):
    config = read_json('./config.json')
    cnn_setting = config['cnn']

    # CNN_Wrapper is used to wrap the model and its training and testing. The function used for cross-validation is
    # CNN_Wrapper.cross_validation.

    with torch.cuda.device(gpu_index):
        cnn = CNN_Wrapper(fil_num=cnn_setting['fil_num'],
                          drop_rate=cnn_setting['drop_rate'],
                          batch_size=cnn_setting['batch_size'],
                          balanced=cnn_setting['balanced'],
                          learn_rate=cnn_setting['learning_rate'],
                          train_epoch=cnn_setting['train_epochs'],
                          dataset=config["dataset"],
                          data_dir=config['Data_dir'],
                          external_dataset=config["external_dataset"],
                          data_dir_ex=config["Data_dir_ex"],
                          seed=config["seed"],
                          model_name='cnn',
                          metric='accuracy',
                          process=process)
        if fold_index == -1:
            cnn.validate()
        else:
            if not os.path.exists("checkpoint_dir/cnn_exp{}".format(fold_index)):
                os.mkdir("checkpoint_dir/cnn_exp{}".format(fold_index))
            cnn.cross_validation(fold_index)


def main(argv):
    if len(argv) != 1:
        print("Error!")
        return
    process = argv[1]

    # Read related parameters
    config = read_json('./config.json')
    folds = config["folds"]
    gpus = config["gpus"]

    # to perform CNN training and testing
    if len(folds) > 0:
        j = 0
        for i in range(len(folds)):
            p = multiprocessing.Process(target=cnn_main, args=(process, gpus[j], folds[i],))
            p.start()
            j += 1
            time.sleep(60)
    else:
        p = multiprocessing.Process(target=cnn_main, args=(process, gpus[0], -1,))


if __name__ == "__main__":
    print("Hello World!")

    main(sys.argv[1:])

    print("Main process end.")
