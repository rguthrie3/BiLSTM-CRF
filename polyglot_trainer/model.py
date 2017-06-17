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
        #return O * dy.rectify(H * rep + Hb) + Ob # ReLU gives less good neighbors

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
 

def wordify(instance):
    return ''.join([i2c[i] for i in instance.chars])
    
def dist(instance, vec):
    we = instance.word_emb
    if options.cosine:
        return 1.0 - (we.dot(vec) / (np.linalg.norm(we) * np.linalg.norm(vec)))
    return np.linalg.norm(we - vec)
 
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
parser.add_argument("--normalized-targets", dest="normalized_targets", action="store_true", help="if toggled, train on normalized vectors from set")
parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="amount of dropout to apply to LSTM part of graph")
parser.add_argument("--num-epochs", default=10, dest="num_epochs", type=int, help="Number of full passes through training set (default = 10)")
parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate")
parser.add_argument("--cosine", dest="cosine", action="store_true", help="Use cosine as diff measure")
parser.add_argument("--dynet-mem", help="Ignore this outside argument")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
options = parser.parse_args()

# Set up logging
log_dir = "embedding_train_charlstm-{}-{}".format(datetime.datetime.now().strftime('%y%m%d%H%M%S'), options.lang)
os.mkdir(log_dir)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_dir + '/log.txt', 'w', 'utf-8')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)

root_logger.info("Training dataset: {}".format(options.dataset))
root_logger.info("Output vocabulary: {}".format(options.vocab))
root_logger.info("Output location: {}\n".format(options.output))

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
root_logger.info("Training Algorithm: {}".format(type(trainer)))

root_logger.info("Number training instances: {}".format(len(training_instances)))

# Create dev set
random.shuffle(training_instances)
dev_cutoff = int(99 * len(training_instances) / 100)
dev_instances = training_instances[dev_cutoff:]
training_instances = training_instances[:dev_cutoff]

if options.debug:
    train_instances = training_instances[:int(len(training_instances)/20)]
    dev_instances = dev_instances[:int(len(dev_instances)/20)]
else:
    train_instances = training_instances

if options.normalized_targets:
    train_instances = [Instance(ins.chars, ins.word_emb/np.linalg.norm(ins.word_emb)) for ins in train_instances]
    dev_instances = [Instance(ins.chars, ins.word_emb/np.linalg.norm(ins.word_emb)) for ins in dev_instances]

epcs = int(options.num_epochs)
pretrained_vec_norms = 0.0
inferred_vec_norms = 0.0
# Shuffle set, divide into cross-folds each epoch
for epoch in xrange(epcs):
    bar = progressbar.ProgressBar()
    
    train_loss = 0.0
    train_correct = Counter()
    train_total = Counter()

    if options.dropout > 0:
        model.set_dropout(options.dropout)
    
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
            word = wordify(instance)
            if word in vocab_words:
                pretrained_vec_norms += np.linalg.norm(instance.word_emb)
                if options.all_from_lstm:
                    vocab_words[word] = np.array(obs_emb.value())
                    inferred_vec_norms += np.linalg.norm(vocab_words[word])
                else: # log vocab embeddings
                    vocab_words[word] = instance.word_emb
        
    root_logger.info("\n")
    root_logger.info("Epoch {} complete".format(epoch + 1))
    trainer.update_epoch(1)
    print trainer.status()
    
    # Evaluate dev data
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
            word = wordify(instance)
            if word in vocab_words:
                pretrained_vec_norms += np.linalg.norm(instance.word_emb)
                if options.all_from_lstm:
                    vocab_words[word] = np.array(obs_emb.value())
                    inferred_vec_norms += np.linalg.norm(vocab_words[word])
                else: # log vocab embeddings
                    vocab_words[word] = instance.word_emb
    
    root_logger.info("Train Loss: {}".format(train_loss))
    root_logger.info("Dev Loss: {}".format(dev_loss))

root_logger.info("\n")
root_logger.info("Average norm for pre-trained in vocab: {}".format(pretrained_vec_norms / len(vocab_words)))

# Infer for test set
showcase_size = 25
top_to_show = 10
showcase = [] # sample for similarity sanity check
for idx, instance in enumerate(test_instances):
    word = wordify(instance)
    obs_emb = model.predict_emb(instance.chars)
    vocab_words[word] = np.array(obs_emb.value())
    inferred_vec_norms += np.linalg.norm(vocab_words[word])
    
    # reservoir sampling
    if idx < showcase_size:
        showcase.append(word)
    else:
        rand = random.randint(0,idx-1)
        if rand < showcase_size:
            showcase[rand] = word
    
root_logger.info("Average norm for trained: {}".format(inferred_vec_norms / len(test_instances)))

similar_words = {}
for w in showcase:
    vec = vocab_words[w]
    top_k = [(wordify(instance),d) for instance,d in sorted([(inst, dist(inst, vec)) for inst in training_instances], key=lambda x: x[1])[:top_to_show]]
    if options.debug:
        print w, [(i,d) for i,d in top_k]
    similar_words[w] = top_k

    
# write all
with codecs.open(options.output, "w", "utf-8") as writer:
    for vw, emb in vocab_words.iteritems():
        writer.write(vw + " ")
        for i in emb:
            writer.write(str(i) + " ")
        writer.write("\n")

# TODO save model

#root_logger.info("\nSome most-similar words from training set for a random selection of test set:\n{}".format("\n".join([k + ":\t" + " ".join([t[0] for t in v]) for k,v in similar_words.iteritems()])))
