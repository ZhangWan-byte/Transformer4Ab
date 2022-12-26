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


def seq_clip(seq, target_length=800, seq_clip_mode=0):
    """clip sequence to target length

    :param seq: seq
    :param target_length: target length, defaults to 800
    :param seq_clip_mode: 0 - random sample / 1 - k nearest amino acids
    :return: clipped sequence
    """
    
    # pad if smaller
    if len(seq) <= target_length:
        subseq = seq + "#" * (target_length - len(seq))
        return subseq
    # random sample if larger
    else:
        if seq_clip_mode==0:
            seq = [(i, seq[i]) for i in range(len(seq))]
            subseq = random.sample(seq, target_length)
            subseq = sorted(subseq, key=lambda x:x[0])
            subseq = "".join([subseq[i][1] for i in range(len(subseq))])
        elif seq_clip_mode==1:
            pass
        else:
            pass

        return subseq

