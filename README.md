Copyright (C) 2022 Xu Tian (tianxu@csu.edu.cn), Jin Liu (liujin06@csu.edu.cn)

## Package Title: 
A Fully Automated MRI-based Multi-task Decoupling Representation Learning for Alzheimer’s Disease Detection and MMSE Score Prediction

## Description:   
This package is designed to enable multi-task decoupled representation learning and automatic diagnosis and MMSE score prediction of AD patients from structural magnetic resonance imaging (sMRI) brain scans.

In this code, "config.json" places the model's hyperparameters and other information. "model.py" places the main CNN model. "model_wrapper.py" encapsulates the training, validation and testing process of the model. "model.py" encapsulates the main CNN model. "dataloader.py" encapsulates the data usage process required by the model. "loss.py" contains forward and backward computations for part of the model's loss.

## How to run this project:
This project must run in python>=2.7, The following steps should be taken to run this project:

1. Data preparation: Before the code can run, the data needs to be prepared. You need to put a file named "dataset.csv" in the "opencsv" folder. "dataset" is the name of the dataset in the configuration file. In this csv files, subjects' AD labels and MMSE scores should be placed in the second and third columns respectively.

2. Environment building:  
    (1) Software: Information about the packages required by the code is at "requirements.txt".   
    (2) Hardware: This code has been tested with NVIDIA GTX2080.
   
3. Code running:    
    (1) Information about code running in "config.json" should be modified.   
    (2) You can use         
    
        python main.py   
      
   to complete the training and testing of the model, or use  

        python main.py train     
        python main.py test
      
   to complete them separately. 
