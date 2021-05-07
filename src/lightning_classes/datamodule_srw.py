#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: romanshen 
@file: datamodule_srw.py 
@time: 2021/05/07
@contact: xiangqing.shen@njust.edu.cn
"""


import logging
import math

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

logger = logging.getLogger(__name__)


class SrwCollator:
    def __init__(self, pretrained_embedding, lang):
        self.pretrained_embedding = load_embeddings(pretrained_embedding, lang)

    def __call__(self, batches):
        pass


def load_embeddings(file, lang):
    emb_matrix = None
    emb_dict = KeyedVectors.load_word2vec_format(file, binary=True)
    for i in range(3, lang.n_words):
        base = math.sqrt(6 / 200)
        word = lang.index2word[i]
        try:
            vector = emb_dict[word]
        except KeyError:
            vector = np.random.uniform(-base, base, 200)
        if np.any(emb_matrix):
            emb_matrix = np.vstack((emb_matrix, vector))
        else:
            emb_matrix = np.random.uniform(-base, base, (4, 200))
            emb_matrix[3] = vector
    return torch.from_numpy(emb_matrix)
