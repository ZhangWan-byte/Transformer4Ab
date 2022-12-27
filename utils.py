import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from Bio import Align


vocab = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 12,
    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'U': 18,
    'V': 19,
    'W': 20,
    'Y': 21,
    '*': 22,    # UNK
    '#': 23,    # PAD / mask
    '/': 24,    # SEP
}


def toOneHot(seq, seq_length=128):
    li = []

    for i in range(len(seq)):
        li.append(vocab[seq[i]])

    if len(li)>seq_length:
        li = li[:seq_length]
    if len(li)<seq_length:
        pad = [vocab["<MASK>"]]*(seq_length-len(li))
        
    return np.array((li+pad))

def to_onehot(seq):
    li = []

    for i in range(len(seq)):
        li.append(vocab[seq[i]])

    return np.array(li)


def seq_sim(target, query):

    aligner = Align.PairwiseAligner()
    aligner = Align.PairwiseAligner(match_score=1.0)

    score = aligner.score(target, query)
    score = score / max(len(target), len(query))
    
    return score


def get_knearest_epi(data, mode=0, K=48, threshold=10):

    """only reserve k nearest amino acids as epitope

    :return: [data_entry]
    """

    # get k nearest (K = 48)
    if mode==0:
        for i in range(len(data)):
            # maintain a heap with k amino acids
            epitope = []
            Apos = np.hstack(data[i]["Apos"])
            Aseq = "".join(data[i]["Aseq"])
            for Aidx in range(len(Aseq)):
                # traverse heavy/light chain to find nearest distance
                nearest_dist = np.inf
                for Hidx in range(len(data[i]["Hpos"])):
                    cur_dist = np.sqrt(np.sum((Apos[Aidx][0] - data[i]["Hpos"][Hidx][0]) ** 2))
                    nearest_dist = np.min([cur_dist, nearest_dist])
                for Lidx in range(len(data[i]["Lpos"])):
                    cur_dist = np.sqrt(np.sum((Apos[Aidx][0] - data[i]["Lpos"][Lidx][0]) ** 2))
                    nearest_dist = np.min([cur_dist, nearest_dist])

                epitope.append((nearest_dist, Aidx))

            epitope_heap = heapq.nsmallest(K, epitope, key=lambda x:x[0])
            epitope_index = sorted([i[1] for i in epitope_heap])

            data[i]["epitope"] = "".join([aseq[i] for i in epitope_index])

    # get within threshold (10 Anstrom)
    if mode==1:
        pass

    return data


def seq_pad_clip(seq, target_length=800):
    """clip sequence to target length

    :param seq: seq
    :param target_length: target length, defaults to 800
    :return: clipped sequence
    """
    
    # padding if smaller
    if len(seq) <= target_length:
        subseq = seq + "#" * (target_length - len(seq))
        return subseq
    else:
        seq = [(i, seq[i]) for i in range(len(seq))]
        subseq = random.sample(seq, target_length)
        subseq = sorted(subseq, key=lambda x:x[0])
        subseq = "".join([subseq[i][1] for i in range(len(subseq))])
        
        return subseq

