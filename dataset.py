import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from utils import *


def get_pair(data, para_seq_length=128, epi_seq_length=800):
    pair_data = []

    for i in range(len(data)):
        # paratope
        paratope = data[i]["Hseq"][0] + "/" + data[i]["Lseq"][0]
        paratope = seq_clip(seq=paratope, target_length=para_seq_length)

        # generate positive sample
        antigen_pos = "/".join(data[i]["Aseq"])
        antigen_pos = seq_clip(seq=antigen_pos, target_length=epi_seq_length)
        
        # generate negative sample
        j = random.randint(0,len(data)-1)
        antigen_neg = "/".join(data[j]["Aseq"])
        while seq_sim(antigen_neg, antigen_pos)>=0.5:
            j = random.randint(0, len(data)-1)
            antigen_neg = "/".join(data[j]["Aseq"])
        antigen_neg = seq_clip(seq=antigen_neg, target_length=epi_seq_length)
        
        # append to pair_data
        pair_data.append((paratope, antigen_pos, 1))
        pair_data.append((paratope, antigen_neg, 0))
        
    return pair_data


# SAbDab
class SAbDabDataset(torch.utils.data.Dataset):
    def __init__(self, data, para_seq_length=128, epi_seq_length=800, kfold=10, holdout_fold=0, is_train=True, is_shuffle=True):
        self.pair_data = get_pair(data, para_seq_length=para_seq_length, epi_seq_length=epi_seq_length)
        if is_shuffle==True:
            random.shuffle(self.pair_data)

        self.is_train = is_train

        self.label = torch.Tensor([pair[-1] for pair in self.pair_data])
        self.data = [(to_onehot(pair[0]), to_onehot(pair[1])) for pair in self.pair_data]

        # train data
        self.data_folds = []
        self.label_folds = []
        for k in range(kfold):
            data_tmp = self.data[k*int(0.1*len(self.data)):(k+1)*int(0.1*len(self.data))]
            label_tmp = self.label[k*int(0.1*len(self.label)):(k+1)*int(0.1*len(self.label))]
            self.data_folds.extend(data_tmp)
            self.label_folds.extend(label_tmp)

        # test data
        self.test_data = self.data_folds.pop(holdout_fold)
        self.test_label = self.label_folds.pop(holdout_fold)
        self.train_data = self.data_folds
        self.train_label = self.label_folds
            
    def __len__(self):
        if self.is_train==True:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def __getitem__(self, idx):
        if self.is_train==True:
            return self.train_data[idx][0], self.train_data[idx][1], self.train_label[idx]
        else:
            return self.test_data[idx][0], self.test_data[idx][1], self.test_label[idx]
        
        
# CoV-AbDab
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="../../MSAI_Project/codes/data/sequence_pairs.json", seq_length=800, kfold=10, holdout_fold=0, is_train=True):
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


if __name__=="__main__":
    # SAbDabDataset
    # data = pickle.load(open("../../MSAI_Project/codes/data/data.json", "rb"))
    # dataset = SAbDabDataset(data)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # t=0
    # for i, (para, epi, label) in enumerate(dataloader):
    #     print(i)
    #     print("para", para)
    #     print("epi", epi)
    #     print("label", label)
    #     t += 1

    #     if t==1:
    #         break

    # SeqDataset
    data = pickle.load(open("../../MSAI_Project/codes/data/data.json", "rb"))
    dataset = SeqDataset(data_path="../data/SARS-SAbDab_Shaun/CoV-AbDab_extract.csv", seq_length=128, \
                    kfold=10, holdout_fold=0, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    t=0
    for i, (para, epi, label) in enumerate(dataloader):
        print(i)
        print("para", para)
        print("epi", epi)
        print("label", label)
        t += 1

        if t==1:
            break