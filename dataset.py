import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from utils import *


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="./CoV-AbDab_extract.csv", seq_length=128, kfold=10, holdout_fold=0, is_train=True):
        self.data_df = pd.read_csv(data_path)
#         self.dataset_length = self.data_df.shape[0]
#         self.label = torch.Tensor(self.data_df["Class"])
        self.is_train = is_train
        self.data = self.data_df.sample(frac=1, random_state=42)

        self.label = torch.Tensor(self.data["Class"])
        self.data = pd.concat([self.data_df["Paratope"].map(toOneHot), \
                               self.data_df["Epitope"].map(toOneHot)], axis=1)

        self.data_folds = []
        self.label_folds = []
        for k in range(kfold):
            data_tmp = self.data[k*int(0.1*self.data.shape[0]):(k+1)*int(0.1*self.data.shape[0])]
            label_tmp = self.label[k*int(0.1*self.label.shape[0]):(k+1)*int(0.1*self.label.shape[0])]
            self.data_folds.append(data_tmp)
            self.label_folds.append(label_tmp)
            
#         print(self.data_folds[0])

        self.test_data = self.data_folds.pop(holdout_fold)
        self.test_label = self.label_folds.pop(holdout_fold)
        self.train_data = pd.concat(self.data_folds)
        self.train_label = torch.hstack(self.label_folds)
            
    def __len__(self):
        if self.is_train==True:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]
    
    def __getitem__(self, idx):
        if self.is_train==True:
            return self.train_data.iloc[idx][0], self.train_data.iloc[idx][1], self.train_label[idx]
        else:
            return self.test_data.iloc[idx][0], self.test_data.iloc[idx][1], self.test_label[idx]
        
        
class SAbDabDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="../../MSAI_Project/codes/data/sequence_pairs.json", seq_length=128):
        self.data_df = pd.read_csv(data_path)
#         self.dataset_length = self.data_df.shape[0]
#         self.label = torch.Tensor(self.data_df["Class"])
        self.is_train = is_train
        self.data = self.data_df.sample(frac=1, random_state=42)

        self.label = torch.Tensor(self.data["Class"])
        self.data = pd.concat([self.data_df["Paratope"].map(toOneHot), \
                               self.data_df["Epitope"].map(toOneHot)], axis=1)

        self.data_folds = []
        self.label_folds = []
        for k in range(kfold):
            data_tmp = self.data[k*int(0.1*self.data.shape[0]):(k+1)*int(0.1*self.data.shape[0])]
            label_tmp = self.label[k*int(0.1*self.label.shape[0]):(k+1)*int(0.1*self.label.shape[0])]
            self.data_folds.append(data_tmp)
            self.label_folds.append(label_tmp)
            
#         print(self.data_folds[0])

        self.test_data = self.data_folds.pop(holdout_fold)
        self.test_label = self.label_folds.pop(holdout_fold)
        self.train_data = pd.concat(self.data_folds)
        self.train_label = torch.hstack(self.label_folds)
            
    def __len__(self):
        if self.is_train==True:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]
    
    def __getitem__(self, idx):
        if self.is_train==True:
            return self.train_data.iloc[idx][0], self.train_data.iloc[idx][1], self.train_label[idx]
        else:
            return self.test_data.iloc[idx][0], self.test_data.iloc[idx][1], self.test_label[idx]