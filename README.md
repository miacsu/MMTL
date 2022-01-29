## MTDRL

This is the code of "A Fully Automated MRI-based Multi-task Decoupling Representation Learning for Alzheimer’s disease Detection and MMSE Prediction: a multi-site validation"

## Requirements

Python 3.6+

requirements.txt

This code has been tested with Pytorch 1.7.0 and NVIDIA GTX2080.

## Data preparation

Before the code can run, the data needs to be prepared. You need to put a file named "dataset.csv" in the "opencsv" folder. "dataset" is the name of the dataset and external validation set in the configuration file. In these csv files, subjects' AD labels and MMSE scores should be placed in the second and third columns, respectively.

## Code

"config.json" places the model's hyperparameters and other information. "model.py" places the main CNN model. "model_wrapper.py" encapsulates the training, validation and testing process of the model. "model.py" encapsulates the main CNN model. "dataloader.py" encapsulates the data usage process required by the model. "loss.py" contains forward and backward computations for part of the model's loss.

## Use this code
You can use "python3 main.py" to use this code.
