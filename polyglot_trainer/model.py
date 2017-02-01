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
        return dy.squared_distance(observation, dy.inputVector(target_rep)) # maybe wrap with dy.sqrt()

    
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
        
# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--vocab", required=True, dest="vocab", help="total vocab to output")
parser.add_argument("--output", required=True, dest="output", help="file with all embeddings")
parser.add_argument("--char-dim", dest="char_dim", help="dimension for character embeddings (default = 20)")
parser.add_argument("--hidden-dim", dest="hidden_dim", help="dimension for LSTM layers (default = 50)")
parser.add_argument("--num-lstm-layers", dest="num_lstm_layers", help="Number of LSTM layers (default = 1)")
parser.add_argument("--all-from-lstm", dest="all_from_lstm", action="store_true", help="if toggled, vectors in original training set are overriden by LSTM-generated vectors")
parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="amount of dropout to apply to LSTM part of graph")
parser.add_argument("--num-epochs", default=20, dest="num_epochs", type=int, help="Number of full passes through training set (default = 20)")
parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate")
parser.add_argument("--dynet-mem", help="Ignore this outside argument")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
options = parser.parse_args()

# Load training set

# Shuffle set, divide into cross-folds each iteration

# Infer for test set, write all (including vocab words in training data, based on options.all_from_lstm)
