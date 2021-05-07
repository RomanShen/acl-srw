import torch

#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: romanshen 
@file: srw.py 
@time: 2021/05/07
@contact: xiangqing.shen@njust.edu.cn
"""

from torch import nn


class SrwModel(nn.Module):
    def __int__(
        self,
        vocab_size,
        char_size,
        char_embedding_dim,
        char_hidden_size,
        word_embedding_dim,
        hidden_dim,
        pos_size,
        pos_embedding_size,
        label_size,
        pattern_hidden_dim,
        pattern_embeddings_dim,
        rule_size,
        max_rule_length,
        lstm_num_layers,
        embed_matrix,
    ):
        super().__init__()
        self.vocab_size = char_size
        self.char_size = char_size
        self.char_embedding_dim = char_embedding_dim
        self.char_hidden_size = char_hidden_size
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.pos_size = pos_size
        self.pos_embedding_size = pos_embedding_size
        self.label_size = label_size
        self.lstm_num_layers = lstm_num_layers
        self.rule_size = rule_size
        self.pattern_embedding_dim = pattern_hidden_dim
        self.pattern_hidden_dim = pattern_hidden_dim
        self.max_rule_length = max_rule_length

        self.pos_embedding = nn.Embedding(self.pos_size, self.pos_embedding_size)
        self.char_embedding = nn.Embedding(self.char_size, self.char_embedding_dim)
        self.word_embedding = nn.Embedding.from_pretrained(embed_matrix)
        self.pattern_embedding = nn.Embedding(
            self.rule_size, self.pattern_embedding_dim
        )

        self.char_lstm = nn.LSTM(
            input_size=self.char_embedding_dim,
            hidden_size=self.char_hidden_size,
            num_layers=self.lstm_num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.encoder_lstm = nn.LSTM(
            input_size=self.word_embedding_dim
            + self.char_hidden_size
            + self.pos_embedding_size,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.query_weight = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_weight = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_weight = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.lb = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.lb2 = nn.Linear(self.hidden_dim, 4)

        self.decoder_lstm = nn.LSTM(
            input_size=self.hidden_dim + self.pattern_embedding_dim,
            hidden_size=self.pattern_hidden_dim,
            num_layers=self.lstm_num_layers,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, sentences, pos, chars, entity, labels, triggers, rules):
        sentence_feature = self.encode_sentences(sentences, pos, chars)
        entity_feature = sentence_feature[entity]
        context_feature = self.entity_attend(sentence_feature, entity_feature)
        for i in range(context_feature.size(0)):
            if i != entity:
                h_t = torch.cat((context_feature[i], entity_feature))
                hidden = torch.tanh(self.lb(h_t))
                out_vector = torch.log_softmax(self.lb2(hidden), dim=-1)
                if i in triggers:
                    label = labels[triggers.index(i)]
                else:
                    label = 0
                nll_loss = nn.functional.nll_loss(out_vector, label)
                if i in triggers and len(rules[triggers.index(i)]) > 1:
                    last_output_embeddings = self.pattern_embedding[0]
                    context = context_feature[i]

    def encode_sentences(self, sentences, pos, chars):
        word_features = self.word_embedding(sentences)
        pos_features = self.pos_embedding(pos)
        char_features = []
        for i in range(sentences.size(1)):
            word = chars[i]
            char_feature = self.encode_char(word)
            char_features.append(char_feature)
        char_features = torch.stack(char_features, dim=0)

        features = torch.cat((word_features, char_features, pos_features), dim=1)
        features, _ = self.encoder_lstm(features)
        return features

    def encode_char(self, word):
        c_seq = self.char_embedding(word)
        c_seq, _ = self.char_lstm(c_seq)
        return c_seq[:, -1]

    def entity_attend(self, sentence_feature, entity_feature):
        keys = self.key_weight(sentence_feature)
        query = self.query_weight(entity_feature)
        value = self.value_weight(sentence_feature)
        s = torch.matmul(query.T, keys)
        s = torch.softmax(s, dim=-1)
        context_feature = torch.mul(value.T, s)
        return context_feature.T  # [seq_len, dim]
