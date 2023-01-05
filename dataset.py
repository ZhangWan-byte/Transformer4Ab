import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *


def set_seed(seed=3407):
    random.seed(3407)
    torch.manual_seed(3407)
    np.random.seed(3407)


def get_pair(data, para_seq_length=128, epi_seq_length=800, seq_clip_mode=1, neg_sample_mode=1, K=48):
    
    """process original data to format in pairs

    :param data: original data
        ['pdb', 'Hchain', 'Lchain', 'Achain', 'Hseq', 'Lseq', 'Aseq', 'L1', 'L2', 'L3', 'H1', 'H2', 'H3', 'Hpos', 'Lpos', 'Apos']
        "Apos": [N, CA, C, O]
    :param para_seq_length: paratope sequence length, defaults to 128
    :param epi_seq_length: epitope sequence length, defaults to 800
    :param seq_clip_mode: padding antigen seq if shorter than L else 0 - random sampling / 1 - k nearest amino acids, defaults to 1
    :param neg_sample_mode: 0-random sampling from dataset / 1 - random sequence / 2 - choose from BLAST, defaults to 1
    :return: [(paratope, antigen_pos, 1), (paratope, antigen_neg, 0), ...]
    """
    
    pair_data = []

    # seq_clip_mode
    # 0 - random sample amino acids
    if seq_clip_mode==0:
        pass
    # 1 - k nearest amino acids
    elif seq_clip_mode==1:
        data = get_knearest_epi(data, K=48)
    else:
        print("Not Implemented seq_clip_mode number!")


    print("Start getting pair data...")
    for i in tqdm(range(len(data))):

        # paratope
        paratope = data[i]["Hseq"][0] + "/" + data[i]["Lseq"][0]
        paratope = seq_pad_clip(seq=paratope, target_length=para_seq_length)

        # epitope - positive sample
        if seq_clip_mode==0:
            antigen_pos = "/".join(data[i]["Aseq"])
            antigen_pos = seq_pad_clip(seq=antigen_pos, target_length=epi_seq_length)
        elif seq_clip_mode==1:
            antigen_pos = data[i]["epitope"]
        else:
            print("Not Implemented seq_clip_mode!")

        # epitope - negative sample
        # 0 - random sample from all epitope seqs
        if seq_clip_mode==0:
            if neg_sample_mode==0:
                j = random.randint(0, len(data)-1)
                antigen_neg = "/".join(data[j]["Aseq"])
                
                # re-sample if sim score >= 0.5
                while seq_sim(antigen_neg, antigen_pos)>=0.9:
                    j = random.randint(0, len(data)-1)
                    antigen_neg = "/".join(data[j]["Aseq"])

                antigen_neg = seq_pad_clip(seq=antigen_neg, target_length=epi_seq_length)
            # 1 - random sequence
            elif neg_sample_mode==1:
                candidates = "".join([k for k in vocab.keys()])
                antigen_neg = "".join(random.choices(candidates, k=epi_seq_length))
            # 2 - BLAST
            else:
                print("Not Implemented BLAST!")
                pass
        elif seq_clip_mode==1:
            if neg_sample_mode==0:
                j = random.randint(0, len(data)-1)
                antigen_neg = data[j]["epitope"]
                
                while seq_sim(antigen_neg, antigen_pos)>=0.5:
                    j = random.randint(0, len(data)-1)
                    antigen_neg = data[j]["epitope"]

                antigen_neg = seq_pad_clip(seq=antigen_neg, target_length=epi_seq_length)
            # 1 - random sequence
            elif neg_sample_mode==1:
                candidates = "".join([k for k in vocab.keys()])
                antigen_neg = "".join(random.choices(candidates, k=epi_seq_length))
            # 2 - BLAST
            else:
                print("Not Implemented BLAST!")
                pass

        # append to pair_data
        pair_data.append((paratope, antigen_pos, 1))
        pair_data.append((paratope, antigen_neg, 0))
        
    return pair_data


# SAbDab
class SAbDabDataset(torch.utils.data.Dataset):
    def __init__(
            self, \
            data, \
            para_seq_length=128, \
            epi_seq_length=800, \
            seq_clip_mode=1, \
            neg_sample_mode=1, \
            kfold=10, \
            holdout_fold=0, \
            is_train=True, \
            is_shuffle=False, \
            folds_path=None, \
            save_path=None,
            K=48
        ):
        # load folds if existing else preprocessing
        if folds_path==None:
            print("folds_path none, preprocessing...")
            self.pair_data = get_pair(data=data, para_seq_length=para_seq_length, epi_seq_length=epi_seq_length, \
                seq_clip_mode=seq_clip_mode, neg_sample_mode=neg_sample_mode, K=K)
            if save_path!=None:
                pickle.dump(self.pair_data, open(save_path, "wb"))
            else:
                pickle.dump(self.pair_data, open("./data/processed_data_clip{}_neg{}.pkl".format(seq_clip_mode, neg_sample_mode), "wb"))
        else:
            print("loading preprocessed data from {}".format(folds_path))
            self.pair_data = pickle.load(open(folds_path, "rb"))

        if is_shuffle==True:
            random.shuffle(self.pair_data)

        self.is_train = is_train

        self.label = torch.Tensor([pair[-1] for pair in self.pair_data])
        self.data = torch.Tensor([[to_onehot(pair[0]), to_onehot(pair[1])] for pair in self.pair_data])

        # train data
        self.data_folds = []
        self.label_folds = []
        for k in range(kfold):
            data_tmp = self.data[k*int((1/kfold)*len(self.data)):(k+1)*int((1/kfold)*len(self.data)), :, :]
            label_tmp = self.label[k*int((1/kfold)*len(self.label)):(k+1)*int((1/kfold)*len(self.label))]
            self.data_folds.append(data_tmp)
            self.label_folds.append(label_tmp)

        # test data
        self.test_data = self.data_folds.pop(holdout_fold)
        self.test_label = self.label_folds.pop(holdout_fold)
        self.train_data = torch.vstack(self.data_folds)
        self.train_label = torch.hstack(self.label_folds)
            
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