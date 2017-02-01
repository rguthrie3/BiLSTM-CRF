from __future__ import division
from collections import Counter
from evaluate_morphotags import Evaluator

import collections
import argparse
import random
import cPickle
import logging
import progressbar
import os
import math
import dynet as dy
import numpy as np

POLYGLOT_UNK = unicode("<UNK>")
PADDING_CHAR = "<*>"

Instance = collections.namedtuple("Instance", ["chars", "word_emb"])

class LSTMPredictor:

    def __init__(self, num_lstm_layers, charset_size, char_dim=20, hidden_dim=50, word_embedding_dim=64, vocab_size=None):
        self.model = dy.Model()
        
        # Char LSTM Parameters
        self.char_lookup = self.model.add_lookup_parameters((charset_size, char_dim))
        self.char_bi_lstm = dy.BiRNNBuilder(num_lstm_layers, char_dim, hidden_dim, self.model, dy.LSTMBuilder)
        
		# Post-LSTM Parameters
		self.lstm_to_rep_params = self.model.add_parameters((word_embedding_dim, hidden_dim * 2))
        self.lstm_to_rep_bias = self.model.add_parameters(word_embedding_dim)
        self.mlp_out = self.model.add_parameters((word_embedding_dim, word_embedding_dim))
        self.mlp_out_bias = self.model.add_parameters(word_embedding_dim)

    def build_graph(self, chars):
        dy.renew_cg()

		pad_char = c2i[PADDING_CHAR]
        char_ids = [pad_char] + chars + [pad_char]
		embeddings = [self.char_lookup[cid] for cid in char_ids]

        bi_lstm_out = self.char_bi_lstm.transduce(embeddings)
		rep = dy.concatenate([bi_lstm_out[0], bi_lstm_out[-1]]) # TODO Make sure this shouldn't be just -1

        H = dy.parameter(self.lstm_to_rep_params)
        Hb = dy.parameter(self.lstm_to_rep_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)
        return O * dy.tanh(H * rep + Hb) + Ob

    def loss(self, chars, target_rep):
        observation = self.build_graph(chars)
        # TODO
        return errors

    
    def set_dropout(self, p):
        self.char_bi_lstm.set_dropout(p)


    def disable_dropout(self):
        self.char_bi_lstm.disable_dropout()

    def save(self, file_name):
        members_to_save = []
        members_to_save.append(self.char_lookup)
        members_to_save.append(self.char_bi_lstm)
        members_to_save.append(self.lstm_to_rep_params)
        members_to_save.append(self.lstm_to_rep_bias)
        members_to_save.append(self.mlp_out)
        members_to_save.append(self.mlp_out_bias)
        self.model.save(file_name, members_to_save)

    @property
    def model(self):
        return self.model