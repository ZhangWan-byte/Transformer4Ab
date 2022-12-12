import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from Bio import Align


vocab = {
    '<MASK>': 0,
    'A': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'O': 13,
    'P': 14,
    'Q': 15,
    'R': 16,
    'S': 17,
    'T': 18,
    'U': 19,
    'V': 20,
    'W': 21,
    'Y': 22,
    '*': 23,    # UNK
    '#': 24,    # PAD
    '/': 25     # SEP
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


def seq_sim(target, query):

    aligner = Align.PairwiseAligner()
    aligner = Align.PairwiseAligner(match_score=1.0)

    score = aligner.score(target, query)
    score = score / max(len(target), len(query))
    
    return score
