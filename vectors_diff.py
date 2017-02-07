from __future__ import division
from collections import Counter

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

import utils


Instance = collections.namedtuple("Instance", ["sentence", "tags"])

NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
POS_KEY = "POS"
PADDING_CHAR = "<*>"


def get_next_att_batch(attributes, att_tuple, idx):
    ret = {}
    for i, att in enumerate(attributes, idx):
        ret[att] = att_tuple[i]
    return ret
    
class BiLSTM_CRF:

    def __init__(self, rnn_model, use_char_rnn):
        self.use_char_rnn = use_char_rnn
        
        self.model = dy.Model()
        att_tuple = self.model.load(rnn_model)
        self.attributes = open(rnn_model + "-atts", "r").read().split("\t")
        att_ct = len(self.attributes)
        idx = 0
        self.words_lookup = att_tuple[idx]
        idx += 1
        if (self.use_char_rnn):
            self.char_lookup = att_tuple[idx]
            idx += 1
            self.char_bi_lstm = att_tuple[idx]
            idx += 1
        self.bi_lstm = att_tuple[idx]
        idx += 1
        self.lstm_to_tags_params = get_next_att_batch(self.attributes, att_tuple, idx)
        idx += att_ct
        self.lstm_to_tags_bias = get_next_att_batch(self.attributes, att_tuple, idx)
        idx += att_ct
        self.mlp_out = get_next_att_batch(self.attributes, att_tuple, idx)
        idx += att_ct
        self.mlp_out_bias = get_next_att_batch(self.attributes, att_tuple, idx)
        idx += att_ct
        self.transitions = get_next_att_batch(self.attributes, att_tuple, idx)    

        # TODO Morpheme embedding parameters
        self.morpheme_lookup = None
        
    
    def set_dropout(self, p):
        self.bi_lstm.set_dropout(p)


    def disable_dropout(self):
        self.bi_lstm.disable_dropout()


    def word_rep(self, word):
        return self.words_lookup[word]
        
    def size(self):
        return self.words_lookup.shape()[0]

    @property
    def model(self):
        return self.model

class LSTMTagger:

    def __init__(self, rnn_model, use_char_rnn):
        self.use_char_rnn = use_char_rnn
        
        self.model = dy.Model()
        att_tuple = self.model.load(rnn_model)
        self.attributes = open(rnn_model + "-atts", "r").read().split("\t")
        att_ct = len(self.attributes)
        idx = 0
        self.words_lookup = att_tuple[idx]
        idx += 1
        if (self.use_char_rnn):
            self.char_lookup = att_tuple[idx]
            idx += 1
            self.char_bi_lstm = att_tuple[idx]
            idx += 1
        self.word_bi_lstm = att_tuple[idx]
        idx += 1
        self.lstm_to_tags_params = get_next_att_batch(self.attributes, att_tuple, idx)
        idx += att_ct
        self.lstm_to_tags_bias = get_next_att_batch(self.attributes, att_tuple, idx)
        idx += att_ct
        self.mlp_out = get_next_att_batch(self.attributes, att_tuple, idx)
        idx += att_ct
        self.mlp_out_bias = get_next_att_batch(self.attributes, att_tuple, idx)


    def word_rep(self, w):
        return self.words_lookup[w]
        
    def size(self):
        return self.words_lookup.shape()[0]

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--embs-dict", dest="embs_dict", default=None, help="Original embeddings dictionary")
parser.add_argument("--model", required=True, dest="model_file", help="Model file to use (.bin)")
parser.add_argument("--model2", dest="model_file2", help="2nd model file to use (.bin)")
parser.add_argument("--viterbi", dest="viterbi", action="store_true", help="Use viterbi training instead of CRF")
parser.add_argument("--no-sequence-model", dest="no_sequence_model", action="store_true", help="Use regular LSTM tagger with no viterbi")
parser.add_argument("--pos-separate-col", default=True, dest="pos_separate_col", help="Output examples have POS in separate column")
parser.add_argument("--use-char-rnn", dest="use_char_rnn", action="store_true", help="Use character RNN")
parser.add_argument("--output", default="compare_output.txt", dest="output", help="Output location")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
options = parser.parse_args()

# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
logging.basicConfig(filename=options.output, filemode="w", format="%(message)s", level=logging.INFO)


# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
if options.viterbi:
    objective = "Viterbi"
elif options.no_sequence_model:
    objective = "No Sequence Model"
else:
    objective = "CRF"
logging.info(
"""
Dataset: {}
Original embeddings: {}
Model input: {}
2nd model input: {}
Objective: {}

""".format(options.dataset, options.embs_dict, options.model_file, options.model_file2, objective))

if options.debug:
    print "DEBUG MODE"

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = cPickle.load(open(options.dataset, "r"))
w2i = dataset["w2i"]

# ===-----------------------------------------------------------------------===
# Load model
# ===-----------------------------------------------------------------------===

tag_set_sizes = { att: len(t2i) for att, t2i in dataset["t2is"].items() }
if options.no_sequence_model:
    model = LSTMTagger(options.model_file, options.use_char_rnn)
else:
    model = BiLSTM_CRF(options.model_file, options.use_char_rnn)
print "Model has {} word embeddings".format(model.size())

if options.embs_dict is not None:
    word_embeddings = utils.read_pretrained_embeddings(options.embs_dict, w2i)
    total_words = len(word_embeddings)
    print "Init dict has {} word embeddings".format(total_words)
else:
    if options.no_sequence_model:
        model2 = LSTMTagger(options.model_file2, options.use_char_rnn)
    else:
        model2 = BiLSTM_CRF(options.model_file2, options.use_char_rnn)
    total_words = model2.size()
    print "Model2 has {} word embeddings".format(total_words)

unfound_words = 0
total_same = 0
total_diff = 0.0
for word in xrange(total_words):
    if options.embs_dict is not None:
        diff = np.linalg.norm(word_embeddings[word] - model.word_rep(word).npvalue())
    else:
        diff = np.linalg.norm(model2.word_rep(word).npvalue() - model.word_rep(word).npvalue())
    if diff > 0.0:
        total_diff += diff
    else:
        total_same += 1

logging.info("Unchanged words: {} of {} = {}".format(total_same, total_words, total_same / total_words))
logging.info("Total diff: {}".format(total_diff))
logging.info("Average diff: {}".format(total_diff / total_words))
print "Total diff: {}".format(total_diff)
print "Average diff: {}".format(total_diff / total_words)
