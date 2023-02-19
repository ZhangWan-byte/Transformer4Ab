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


def get_random_sequence(length=48):
    candidates = "".join([k for k in vocab.keys()])
    candidates = candidates[:-3]
    antigen_neg = "".join(random.choices(candidates, k=length))

    return antigen_neg


def get_pair(data, epi_seq_length=800, seq_clip_mode=1, neg_sample_mode=1, K=48, use_cache=False, use_pair=False):
    
    """process original data to format in pairs

    :param data: original data
        ['pdb', 'Hchain', 'Lchain', 'Achain', 'Hseq', 'Lseq', 'Aseq', 'L1', 'L2', 'L3', 'H1', 'H2', 'H3', 'Hpos', 'Lpos', 'Apos']
        "Apos": [N, CA, C, O]
    :param epi_seq_length: epitope sequence length, defaults to 800
    :param seq_clip_mode: padding antigen seq if shorter than L else 0 - random sampling / 1 - k nearest amino acids, defaults to 1
    :param neg_sample_mode: 0-random sampling from dataset / 1 - random sequence / 2 - choose from BLAST, defaults to 1
    :return: [(paratope, antigen_pos, 1), (paratope, antigen_neg, 0), ...]
    :return: [(paratope, antigen_pos, antigen_neg)]
    """
    
    pair_data = []

    # seq_clip_mode
    # 0 - random sample amino acids
    if seq_clip_mode==0:
        pass
    # 1 - k nearest amino acids
    elif seq_clip_mode==1:
        if use_cache==False:
            data = get_knearest_epi(data, K=K)
            pickle.dump(data, open("./data/tmp_knnepi.pkl", "wb"))
        else:
            print("loading ./data/tmp_knnepi.pkl as data containing knn epitope")
            data = pickle.load(open("./data/tmp_knnepi.pkl", "rb"))
    else:
        print("Not Implemented seq_clip_mode number!")


    print("Start getting pair data...")
    print("seq_clip_mode: {}\tneg_sample_mode: {}\tuse_pair: {}\t".format(seq_clip_mode, neg_sample_mode, use_pair))
    for i in tqdm(range(len(data))):

        # paratope
        # paratope = data[i]["Hseq"][0] + "/" + data[i]["Lseq"][0]
        paratope = "/".join([data[i]["H1"], data[i]["H2"], data[i]["H3"], data[i]["L1"], data[i]["L2"], data[i]["L3"]])

        # epitope - positive sample
        if seq_clip_mode==0:
            antigen_pos = "/".join(data[i]["Aseq"])
            antigen_pos = seq_pad_clip(seq=antigen_pos, target_length=epi_seq_length)
        elif seq_clip_mode==1:
            antigen_pos = data[i]["epitope"]
            antigen_pos = seq_pad_clip(seq=antigen_pos, target_length=epi_seq_length)
        else:
            print("Not Implemented seq_clip_mode!")

        # epitope - negative sample
        # 0 - random sample amino acids
        if seq_clip_mode==0:
            # 0 - sample from all epitope seqs
            if neg_sample_mode==0:
                j = random.randint(0, len(data)-1)
                antigen_neg = "/".join(data[j]["Aseq"])
                
                # re-sample if sim score >= 0.9
                while seq_sim(antigen_neg, antigen_pos)>=0.5:
                    j = random.randint(0, len(data)-1)
                    antigen_neg = "/".join(data[j]["Aseq"])

                antigen_neg = seq_pad_clip(seq=antigen_neg, target_length=epi_seq_length)
            # 1 - random sequence
            elif neg_sample_mode==1:
                # candidates = "".join([k for k in vocab.keys()])
                # antigen_neg = "".join(random.choices(candidates, k=epi_seq_length))
                antigen_neg = get_random_sequence(length=epi_seq_length)
                antigen_neg = seq_pad_clip(seq=antigen_neg, target_length=epi_seq_length)
            # 2 - BLAST
            else:
                print("Not Implemented BLAST!")
                pass
        # 1 - k nearest amino acids
        elif seq_clip_mode==1:
            # 0 - sample from all epitope seqs
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

        # append to pair_data after removing redundant samples
        # redundancy - 1. >=90%sim for paratope; 2. >=90%sim for epitope; 3. same_label

        if use_pair==False:
            redundant_pos = False
            redundant_neg = False
            for i in range(len(pair_data)):
                if seq_sim(pair_data[i][0], paratope)>=0.9 and seq_sim(pair_data[i][1], antigen_pos)>=0.9 and pair_data[i][2]==1:
                    redundant_pos = True
                    break
                if seq_sim(pair_data[i][0], paratope)>=0.9 and seq_sim(pair_data[i][1], antigen_neg)>=0.9 and pair_data[i][2]==0:
                    redundant_neg = True
                    break
            if redundant_pos==False:
                pair_data.append((paratope, antigen_pos, 1))
            if redundant_neg==False:
                pair_data.append((paratope, antigen_neg, 0))
        else:
            redundant = False
            for i in range(len(pair_data)):
                if seq_sim(pair_data[i][0], paratope)>=0.9 and \
                   seq_sim(pair_data[i][1], antigen_pos)>=0.9 and \
                   seq_sim(pair_data[i][2], antigen_neg)>=0.9:
                    redundant = True
            if redundant==False:
                pair_data.append((paratope, antigen_pos, antigen_neg))
        
    return pair_data


def my_pad_sequence(seqs):
    max_len = max(list(map(lambda x:len(x), seqs)))
    
    seqs = list(map(lambda x:"+"+x.strip("#")+"#"*(max_len-len(x.strip("#")))+"-", seqs))
    
    return seqs

def augment_fn(seq):
    # left-right flipping
    if random.random()<=0.5:
        return seq[::-1]
    else:
        return seq


def collate_fn(batch, mode=0, use_augment=False):

    paras = [b[0] for b in batch]
    epis = [b[1] for b in batch]

    if use_augment==True:
        paras = list(map(augment_fn, paras))
        epis = list(map(augment_fn, epis))

    # +ABCD-###
    if mode==0:
        labels = torch.hstack([b[2] for b in batch])
        max_len = max(max(list(map(lambda x:len(x), paras))), max(list(map(lambda x:len(x), epis))))

        paras = ["+"+i.strip("#")+"-"+"#"*(max_len-len(i.strip("#"))) for i in paras]
        epis = ["+"+i.strip("#")+"-"+"#"*(max_len-len(i.strip("#"))) for i in epis]

        new_batch = [paras, epis, labels]

        return new_batch
    
    # padding for six CDRs
    if mode==1:
        paras = [(p.split("/"), max(list(map(lambda x:len(x), p.split("/"))))) for p in paras]
        paras = list(map(my_pad_sequence, paras))
        labels = [b[2] for b in batch]
        new_batch = [paras, epis, labels]

        return new_batch
    
def pair_collate_fn(batch, mode=0):

    paras = [b[0] for b in batch]
    epis_pos = [b[1] for b in batch]
    epis_neg = [b[2] for b in batch]


    # +ABCD-###
    if mode==0:
        max_len = max(max(list(map(lambda x:len(x), paras))), 
                      max(list(map(lambda x:len(x), epis_pos))), 
                      max(list(map(lambda x:len(x), epis_neg))))

        paras = ["+"+i.strip("#")+"-"+"#"*(max_len-len(i.strip("#"))) for i in paras]
        epis_pos = ["+"+i.strip("#")+"-"+"#"*(max_len-len(i.strip("#"))) for i in epis_pos]
        epis_neg = ["+"+i.strip("#")+"-"+"#"*(max_len-len(i.strip("#"))) for i in epis_neg]

        new_batch = [paras, epis_pos, epis_neg]

        return new_batch


def my_collate_fn1(batch):
    return collate_fn(batch, mode=0, use_augment=False)

def my_collate_fn2(batch):
    return collate_fn(batch, mode=0, use_augment=True)


# SAbDab
class SAbDabDataset(torch.utils.data.Dataset):
    def __init__(
            self, \
            data, \
            epi_seq_length=800, \
            seq_clip_mode=1, \
            neg_sample_mode=1, \
            kfold=10, \
            holdout_fold=0, \
            is_train_test_full="train", \
            is_shuffle=False, \
            folds_path=None, \
            save_path=None, \
            K=48, \
            data_augment=False, \
            augment_ratio=0.5, \
            use_cache=False, \
            use_pair=False
        ):
        # load folds if existing else preprocessing
        if folds_path==None:
            print("folds_path none, preprocessing...")
            self.pair_data = get_pair(data=data, epi_seq_length=epi_seq_length, \
                seq_clip_mode=seq_clip_mode, neg_sample_mode=neg_sample_mode, K=K, use_pair=use_pair, \
                use_cache=use_cache)
            if save_path!=None:
                pickle.dump(self.pair_data, open(save_path, "wb"))
            else:
                print("save to ./data/processed_data_clip{}_neg{}_usepair{}.pkl".format(seq_clip_mode, neg_sample_mode, use_pair))
                pickle.dump(self.pair_data, open("./data/processed_data_clip{}_neg{}_usepair{}.pkl".format(seq_clip_mode, neg_sample_mode, use_pair), "wb"))
        else:
            print("loading preprocessed data from {}".format(folds_path))
            self.pair_data = pickle.load(open(folds_path, "rb"))

        if is_shuffle==True:
            random.shuffle(self.pair_data)

        self.is_train_test_full = is_train_test_full
        self.use_pair = use_pair

        if use_pair==False:
            self.label = torch.Tensor([pair[-1] for pair in self.pair_data])
            self.data = [(pair[0], pair[1]) for pair in self.pair_data]


        if self.is_train_test_full=="train" or self.is_train_test_full=="test":

            # train data
            self.data_folds = []
            self.label_folds = []
            for k in range(kfold):
                data_tmp = self.data[k*int((1/kfold)*len(self.data)):(k+1)*int((1/kfold)*len(self.data))]
                label_tmp = self.label[k*int((1/kfold)*len(self.label)):(k+1)*int((1/kfold)*len(self.label))]
                self.data_folds.append(data_tmp)
                self.label_folds.append(label_tmp)

            # test data
            self.test_data = self.data_folds.pop(holdout_fold)
            self.test_label = self.label_folds.pop(holdout_fold)
            self.train_data = []
            for i in range(len(self.data_folds)):
                for j in range(len(self.data_folds[i])):
                    self.train_data.append(self.data_folds[i][j])
            self.train_label = torch.hstack(self.label_folds)

            # data augmentation
            if data_augment==True:
                print(int(augment_ratio*len(self.train_data)))
                tmp = self.train_data[:int(augment_ratio*len(self.train_data))]
                tmp = [(entry[0], get_random_sequence(length=epi_seq_length)) for entry in tmp]
                # print(len(tmp), tmp)
                # print(self.data.shape, self.data)
                self.train_data = self.train_data + tmp
                self.train_label = torch.hstack([self.train_label, torch.Tensor([0]*int(augment_ratio*len(self.train_data)))])
            
    def __len__(self):
        if self.use_pair==False:
            if self.is_train_test_full=="train":
                return len(self.train_data)
            elif self.is_train_test_full=="test":
                return len(self.test_data)
            else:
                return len(self.data)
        else:
            return len(self.pair_data)
    
    def __getitem__(self, idx):
        if self.use_pair==False:
            if self.is_train_test_full=="train":
                return self.train_data[idx][0], self.train_data[idx][1], self.train_label[idx]
            elif self.is_train_test_full=="test":
                return self.test_data[idx][0], self.test_data[idx][1], self.test_label[idx]
            else:
                return self.data[idx][0], self.data[idx][1], self.label[idx]
        else:
            return self.pair_data[idx][0], self.pair_data[idx][1], self.pair_data[idx][2]
        
        
# CoV-AbDab
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_path="../../MSAI_Project/codes/data/sequence_pairs.json", 
                 kfold=10, 
                 holdout_fold=0, 
                 is_train_test_full="train"):

        self.data_df = pd.read_csv(data_path)
        self.is_train_test_full = is_train_test_full
        self.data = self.data_df.sample(frac=1, random_state=42)

        self.label = torch.Tensor(self.data["Class"])
        # self.data = pd.concat([self.data_df["Paratope"].map(toOneHot), \
        #                        self.data_df["Epitope"].map(toOneHot)], axis=1)
        self.data = pd.concat([self.data_df["Paratope"], \
                               self.data_df["Epitope"]], axis=1)

        if self.is_train_test_full=="train" or self.is_train_test_full=="test":
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
        
        if self.use_pair==True:
            pass

    def __len__(self):
        if self.is_train_test_full=="train":
            return self.train_data.shape[0]
        elif self.is_train_test_full=="test":
            return self.test_data.shape[0]
        else:
            return self.data.shape[0]
    
    def __getitem__(self, idx):
        if self.is_train_test_full=="train":
            return self.train_data.iloc[idx][0], self.train_data.iloc[idx][1], self.train_label[idx]
        elif self.is_train_test_full=="test":
            return self.test_data.iloc[idx][0], self.test_data.iloc[idx][1], self.test_label[idx]
        else:
            return self.data.iloc[idx][0], self.data.iloc[idx][1], self.label[idx]


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