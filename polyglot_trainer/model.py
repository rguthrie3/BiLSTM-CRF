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
import datetime
import codecs
import dynet as dy
import numpy as np

POLYGLOT_UNK = unicode("<UNK>")
PADDING_CHAR = "<*>"

DEFAULT_CHAR_DIM = 20
DEFAULT_HIDDEN_DIM = 50

Instance = collections.namedtuple("Instance", ["chars", "word_emb"])

class LSTMPredictor:

    def __init__(self, num_lstm_layers, charset_size, char_dim, hidden_dim, word_embedding_dim, vocab_size=None):
        self.model = dy.Model()
        
        # Char LSTM Parameters
        self.char_lookup = self.model.add_lookup_parameters((charset_size, char_dim))
        self.char_fwd_lstm = dy.LSTMBuilder(num_lstm_layers, char_dim, hidden_dim, self.model)
        self.char_bwd_lstm = dy.LSTMBuilder(num_lstm_layers, char_dim, hidden_dim, self.model)
        
        # Post-LSTM Parameters
        self.lstm_to_rep_params = self.model.add_parameters((word_embedding_dim, hidden_dim * 2))
        self.lstm_to_rep_bias = self.model.add_parameters(word_embedding_dim)
        self.mlp_out = self.model.add_parameters((word_embedding_dim, word_embedding_dim))
        self.mlp_out_bias = self.model.add_parameters(word_embedding_dim)

    def predict_emb(self, chars):
        dy.renew_cg()
        
        finit = self.char_fwd_lstm.initial_state()
        binit = self.char_bwd_lstm.initial_state()
        
        H = dy.parameter(self.lstm_to_rep_params)
        Hb = dy.parameter(self.lstm_to_rep_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)

        pad_char = c2i[PADDING_CHAR]
        char_ids = [pad_char] + chars + [pad_char]
        embeddings = [self.char_lookup[cid] for cid in char_ids]

        bi_fwd_out = finit.transduce(embeddings)
        bi_bwd_out = binit.transduce(reversed(embeddings))
        
        rep = dy.concatenate([bi_fwd_out[-1], bi_bwd_out[-1]])

        return O * dy.tanh(H * rep + Hb) + Ob

    def loss(self, observation, target_rep):
        return dy.squared_distance(observation, dy.inputVector(target_rep))
    
    def set_dropout(self, p):
        self.char_fwd_lstm.set_dropout(p)
        self.char_bwd_lstm.set_dropout(p)

    def disable_dropout(self):
        self.char_fwd_lstm.disable_dropout()
        self.char_bwd_lstm.disable_dropout()

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
parser.add_argument("--lang", dest="lang", default="en", help="language")
parser.add_argument("--char-dim", default=DEFAULT_CHAR_DIM, dest="char_dim", help="dimension for character embeddings (default = 20)")
parser.add_argument("--hidden-dim", default=DEFAULT_HIDDEN_DIM, dest="hidden_dim", help="dimension for LSTM layers (default = 50)")
parser.add_argument("--num-lstm-layers", default=1, dest="num_lstm_layers", help="Number of LSTM layers (default = 1)")
parser.add_argument("--all-from-lstm", dest="all_from_lstm", action="store_true", help="if toggled, vectors in original training set are overriden by LSTM-generated vectors")
parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="amount of dropout to apply to LSTM part of graph")
parser.add_argument("--num-epochs", default=10, dest="num_epochs", type=int, help="Number of full passes through training set (default = 10)")
parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate")
parser.add_argument("--dynet-mem", help="Ignore this outside argument")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
options = parser.parse_args()

# Set up logging
log_dir = "embedding_train_charlstm-{}-{}".format(datetime.datetime.now().strftime('%y%m%d%H%M%S'), options.lang)
os.mkdir(log_dir)
logging.basicConfig(filename=log_dir + "/log.txt", filemode="w", format="%(message)s", level=logging.INFO)

# Load training set
dataset = cPickle.load(open(options.dataset, "r"))
c2i = dataset["c2i"]
i2c = { i: c for c, i in c2i.items() } # inverse map
training_instances = dataset["training_instances"]
test_instances = dataset["test_instances"]
emb_dim = len(training_instances[0].word_emb)

# Load words to write
vocab_words = {}
with codecs.open(options.vocab, "r", "utf-8") as vocab_file:
    for vw in vocab_file.readlines():
        vocab_words[vw.strip()] = np.array([0])

model = LSTMPredictor(options.num_lstm_layers, len(c2i), options.char_dim, options.hidden_dim, emb_dim, vocab_size=None)
trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9, 0.1)
logging.info("Training Algorithm: {}".format(type(trainer)))

logging.info("Number training instances: {}".format(len(training_instances)))

epcs = int(options.num_epochs)
pretrained_vec_norms = 0.0
inferred_vec_norms = 0.0
# Shuffle set, divide into cross-folds each epoch
for epoch in xrange(epcs):
    bar = progressbar.ProgressBar()
    random.shuffle(training_instances)
    
    # random 10% fold for validation
    dev_cutoff = int(9 * len(training_instances) / 10)
    dev_instances = training_instances[dev_cutoff:]
    train_instances = training_instances[:dev_cutoff]
    train_loss = 0.0
    train_correct = Counter()
    train_total = Counter()

    if options.dropout > 0:
        model.set_dropout(options.dropout)

    if options.debug:
        train_instances = train_instances[:int(len(training_instances)/20)]
        dev_instances = dev_instances[:int(len(dev_instances)/20)]
    else:
        train_instances = train_instances
    
    for instance in bar(train_instances):
        if len(instance.chars) <= 0: continue
        obs_emb = model.predict_emb(instance.chars)
        loss_expr = model.loss(obs_emb, instance.word_emb)
        loss = loss_expr.scalar_value()
        
        # Bail if loss is NaN
        if math.isnan(loss):
            assert False, "NaN occured"
            
        train_loss += loss

        # Do backward pass and update parameters
        loss_expr.backward()
        trainer.update()
        
        if epoch == epcs - 1:
            word = ''.join([i2c[i] for i in instance.chars])
            if word in vocab_words:
                pretrained_vec_norms += np.linalg.norm(instance.word_emb)
                if options.all_from_lstm:
                    vocab_words[word] = np.array(obs_emb.value())
                    inferred_vec_norms += np.linalg.norm(vocab_words[word])
                else: # log vocab embeddings
                    vocab_words[word] = instance.word_emb
        
    logging.info("\n")
    logging.info("Epoch {} complete".format(epoch + 1))
    trainer.update_epoch(1)
    print trainer.status()
    
    # Evaluate dev data (remember it's not the same set each epoch)
    model.disable_dropout()
    dev_loss = 0.0
    dev_correct = Counter()
    dev_total = Counter()
    
    bar = progressbar.ProgressBar()
    for instance in bar(dev_instances):
        if len(instance.chars) <= 0: continue
        obs_emb = model.predict_emb(instance.chars)
        dev_loss += model.loss(obs_emb, instance.word_emb).scalar_value()
        
        if epoch == epcs - 1:
            word = ''.join([i2c[i] for i in instance.chars])
            if word in vocab_words:
                pretrained_vec_norms += np.linalg.norm(instance.word_emb)
                if options.all_from_lstm:
                    vocab_words[word] = np.array(obs_emb.value())
                    inferred_vec_norms += np.linalg.norm(vocab_words[word])
                else: # log vocab embeddings
                    vocab_words[word] = instance.word_emb
    
    logging.info("Train Loss: {}".format(train_loss))
    logging.info("Dev Loss: {}".format(dev_loss))

logging.info("\n")
logging.info("Average norm for pre-trained in vocab: {}".format(pretrained_vec_norms / len(training_instances)))

# Infer for test set
bar = progressbar.ProgressBar()
for instance in bar(test_instances):
    word = ''.join([i2c[i] for i in instance.chars])
    obs_emb = model.predict_emb(instance.chars)
    vocab_words[word] = np.array(obs_emb.value())
    inferred_vec_norms += np.linalg.norm(vocab_words[word])
    
logging.info("Average norm for trained: {}".format(inferred_vec_norms / len(test_instances)))

# write all
with codecs.open(options.output, "w", "utf-8") as writer:
    for vw, emb in vocab_words.iteritems():
        writer.write(vw + " ")
        for i in emb:
            writer.write(str(i) + " ")
        writer.write("\n")
