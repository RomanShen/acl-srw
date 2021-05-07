#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: romanshen 
@file: dataset_srw.py 
@time: 2021/05/07
@contact: xiangqing.shen@njust.edu.cn
"""

import pickle
import logging

from torch.utils.data.dataset import Dataset

from prep_utils import Lang


logger = logging.getLogger(__name__)


class SrwDataset(Dataset):
    def __init__(self, data_dir):
        self.input_lang = Lang("input")
        self.pl1 = Lang("position")
        self.char = Lang("char")
        self.rule_lang = Lang("rule")
        self.raw = list()

        (
            self.input_lang,
            self.pl1,
            self.char,
            self.rule_lang,
            self.raw,
        ) = pickle.load(open(data_dir, "rb"))

    def __getitem__(self, index):
        return self.convert_examples_to_features(index)

    def convert_examples_to_features(self, index):
        datapoint = self.raw[index]
        batch = {}
        if datapoint[3][0] != -1:
            batch["input_text"] = [
                self.input_lang.word2index[w] for w in datapoint[0]
            ] + [1]
            batch["entity"] = datapoint[1]
            batch["entity_position"] = datapoint[2]
            batch["trigger_position"] = datapoint[3]
            batch["trigger_label"] = [self.input_lang.label2id[l] for l in datapoint[4]]
            batch["relative_position"] = [
                self.pl1.word2index[p] for p in datapoint[5]
            ] + [0]
            batch["char"] = [
                [self.char.word2index[c] for c in w] for w in datapoint[0] + ["EOS"]
            ]
            batch["rule"] = [
                [self.rule_lang.word2index[p] for p in rule + ["EOS"]]
                for rule in datapoint[6]
            ]
        else:
            batch["input_text"] = [
                self.input_lang.word2index[w] for w in datapoint[0]
            ] + [1]
            batch["entity"] = datapoint[1]
            batch["entity_position"] = datapoint[2]
            batch["trigger_position"] = datapoint[3]
            batch["trigger_label"] = [0]
            batch["relative_position"] = [
                self.pl1.word2index[p] for p in datapoint[5]
            ] + [0]
            batch["char"] = [
                [self.char.word2index[c] for c in w] for w in datapoint[0] + ["EOS"]
            ]
            batch["rule"] = [self.rule_lang.word2index["EOS"]]
        return batch
