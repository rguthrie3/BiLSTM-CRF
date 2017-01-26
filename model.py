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

import utils


Instance = collections.namedtuple("Instance", ["sentence", "tags"])

NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
POS_KEY = "POS"
PADDING_CHAR = "<*>"

DEFAULT_WORD_EMBEDDING_SIZE = 64

class BiLSTM_CRF:

    def __init__(self, tagset_sizes, num_lstm_layers, hidden_dim, word_embeddings, use_char_rnn, charset_size, train_vocab_ctr, margins, lowercase_words, vocab_size=None):
        self.model = dy.Model()
        self.tagset_sizes = tagset_sizes
        self.train_vocab_ctr = train_vocab_ctr
        self.margins = margins
        self.lowercase_words = lowercase_words

        # Word embedding parameters
        if word_embeddings is not None: # Use pretrained embeddings
            vocab_size = word_embeddings.shape[0]
            word_embedding_dim = word_embeddings.shape[1]
            self.words_lookup = self.model.add_lookup_parameters((vocab_size, word_embedding_dim))
            self.words_lookup.init_from_array(word_embeddings)
        else:
            word_embedding_dim = DEFAULT_WORD_EMBEDDING_SIZE
            self.words_lookup = self.model.add_lookup_parameters((vocab_size, word_embedding_dim))
        
        # Char LSTM Parameters
        self.use_char_rnn = use_char_rnn
        if use_char_rnn:
            self.char_lookup = self.model.add_lookup_parameters((charset_size, 20))
            self.char_bi_lstm = dy.BiRNNBuilder(1, 20, 128, self.model, dy.LSTMBuilder)

        # Word LSTM parameters
        if use_char_rnn:
            input_dim = word_embedding_dim + 128
        else:
            input_dim = word_embedding_dim
        self.bi_lstm = dy.BiRNNBuilder(num_lstm_layers, input_dim, hidden_dim, self.model, dy.LSTMBuilder)
        
        self.attributes = tagset_sizes.keys()
        self.lstm_to_tags_params = {}
        self.lstm_to_tags_bias = {}
        self.mlp_out = {}
        self.mlp_out_bias = {}
        self.transitions = {}
        for attribute, set_size in tagset_sizes.items():
            # Matrix that maps from Bi-LSTM output to num tags
            self.lstm_to_tags_params[attribute] = self.model.add_parameters((set_size, hidden_dim))
            self.lstm_to_tags_bias[attribute] = self.model.add_parameters(set_size)
            self.mlp_out[attribute] = self.model.add_parameters((set_size, set_size))
            self.mlp_out_bias[attribute] = self.model.add_parameters(set_size)
    
            # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
            self.transitions[attribute] = self.model.add_lookup_parameters((set_size, set_size))


    def set_dropout(self, p):
        self.bi_lstm.set_dropout(p)


    def disable_dropout(self):
        self.bi_lstm.disable_dropout()


    def word_rep(self, word):
        '''
        :param word: index of word in lookup table
        '''
        if self.lowercase_words:
            lower_word_form = i2w[word].lower()
            if lower_word_form in w2i:
                word_in_ds = w2i[lower_word_form]
            else:
                word_in_ds = word
        else:
            word_in_ds = word
        wemb = self.words_lookup[word_in_ds]
        if self.use_char_rnn:
            pad_char = c2i[PADDING_CHAR]
            # Note: use original casing ("word") for characters
            char_ids = [pad_char] + [c2i[c] for c in i2w[word]] + [pad_char] # TODO optimize
            char_embs = [self.char_lookup[cid] for cid in char_ids]
            char_exprs = self.char_bi_lstm.transduce(char_embs)
            return dy.concatenate([ wemb, char_exprs[-1] ])
        else:
            return wemb


    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        embeddings = [self.word_rep(w) for w in sentence]
        
        lstm_out = self.bi_lstm.transduce(embeddings)
        
        scores = {}
        H = {}
        Hb = {}
        O = {}
        Ob = {}
        for att in self.attributes:
            H[att] = dy.parameter(self.lstm_to_tags_params[att])
            Hb[att] = dy.parameter(self.lstm_to_tags_bias[att])
            O[att] = dy.parameter(self.mlp_out[att])
            Ob[att] = dy.parameter(self.mlp_out_bias[att])
            scores[att] = []
            for rep in lstm_out:
                score_t = O[att] * dy.tanh(H[att] * rep + Hb[att]) + Ob[att]
                scores[att].append(score_t)

        return scores


    def score_sentence(self, observations, tags, att):
        t2i = t2is[att]
        if len(tags) == 0:
            tags = [t2i[NONE_TAG]] * len(observations)
        assert len(observations) == len(tags)
        trans = self.transitions[att]
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [t2i[START_TAG]] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(trans[tags[i+1]], tags[i]) + dy.pick(obs, tags[i+1])
            score_seq.append(score.value())
        score = score + dy.pick(trans[t2i[END_TAG]], tags[-1])
        return score


    def viterbi_loss(self, sentence, tags_set, use_margins=True):
        observations_set = self.build_tagging_graph(sentence)
        losses = {}
        ret_tags = {}
        for att, observations in observations_set.items():
            tags = tags_set[att]
            viterbi_tags, viterbi_score = self.viterbi_decoding(observations, tags, att, use_margins)
            if viterbi_tags != tags:
                gold_score = self.score_sentence(observations, tags, att)
                losses[att] = viterbi_score - gold_score
                ret_tags[att] = viterbi_tags
            else:
                losses[att] = dy.scalarInput(0)
                ret_tags[att] = viterbi_tags
        return losses, ret_tags


    def neg_log_loss(self, sentence, tags):
        observations_set = self.build_tagging_graph(sentence)
        scores = {}
        for att, observations in observations_set.items():
            gold_score = self.score_sentence(observations, tags[att], att)
            forward_score = self.forward(observations, att)
            scores[att] = forward_score - gold_score
        return scores


    def forward(self, observations, att):

        def log_sum_exp(scores, tagset_size):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * tagset_size)
            return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))

        t2i = t2is[att]
        trans = self.transitions[att]
        tagset_size = self.tagset_sizes[att]
        init_alphas = [-1e10] * tagset_size
        init_alphas[t2i[START_TAG]] = 0
        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(tagset_size):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * tagset_size)
                next_tag_expr = for_expr + trans[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr, tagset_size))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + trans[t2i[END_TAG]]
        alpha = log_sum_exp(terminal_expr, tagset_size)
        return alpha


    def viterbi_decoding(self, observations, gold_tags, att, use_margins):
        t2i = t2is[att]
        tagset_size = self.tagset_sizes[att]
        backpointers = []
        init_vvars   = [-1e10] * tagset_size
        init_vvars[t2i[START_TAG]] = 0 # <Start> has all the probability
        for_expr     = dy.inputVector(init_vvars)
        trans_exprs  = [self.transitions[att][idx] for idx in range(tagset_size)]
        for gold, obs in zip(gold_tags, observations):
            bptrs_t = []
            vvars_t = []
            for next_tag in range(tagset_size):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id  = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs
            if use_margins and self.margins[att] != 0:
                adjust = [self.margins[att]] * tagset_size
                adjust[gold] = 0
                for_expr = for_expr + dy.inputVector(adjust)
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[t2i[END_TAG]]
        terminal_arr  = terminal_expr.npvalue()
        best_tag_id   = np.argmax(terminal_arr)
        path_score    = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id] # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop() # Remove the start symbol
        best_path.reverse()
        assert start == t2i[START_TAG]
        # Return best path and best path's score
        return best_path, path_score

    def save(self, file_name):
        members_to_save = []
        members_to_save.append(self.words_lookup)
        if (self.use_char_rnn):
            members_to_save.append(self.char_lookup)
            members_to_save.append(self.char_bi_lstm)
        members_to_save.append(self.bi_lstm)
        members_to_save.extend(utils.sortvals(self.lstm_to_tags_params))
        members_to_save.extend(utils.sortvals(self.lstm_to_tags_bias))
        members_to_save.extend(utils.sortvals(self.mlp_out))
        members_to_save.extend(utils.sortvals(self.mlp_out_bias))
        members_to_save.extend(utils.sortvals(self.transitions))
        self.model.save(file_name, members_to_save)
        
        with open(file_name + "-atts", 'w') as attdict:
            attdict.write("\t".join(sorted(self.attributes)))
    
    @property
    def model(self):
        return self.model


class LSTMTagger:

    def __init__(self, tagset_sizes, num_lstm_layers, hidden_dim, word_embeddings, train_vocab_ctr, use_char_rnn, charset_size, lowercase_words, att_props=None, vocab_size=None, word_embedding_dim=None):
        self.model = dy.Model()
        self.tagset_sizes = tagset_sizes
        self.train_vocab_ctr = train_vocab_ctr
        self.attributes = tagset_sizes.keys()
        self.lowercase_words = lowercase_words
        if att_props is not None:
            self.att_props = {att:(1.0-p) for att,p in att_props.iteritems()}
        else:
            self.att_props = None
        
        if word_embeddings is not None: # Use pretrained embeddings
            vocab_size = word_embeddings.shape[0]
            word_embedding_dim = word_embeddings.shape[1]
            self.words_lookup = self.model.add_lookup_parameters((vocab_size, word_embedding_dim))
            self.words_lookup.init_from_array(word_embeddings)
        else:
            self.words_lookup = self.model.add_lookup_parameters((vocab_size, word_embedding_dim))

        # Char LSTM Parameters
        self.use_char_rnn = use_char_rnn
        if use_char_rnn:
            self.char_lookup = self.model.add_lookup_parameters((charset_size, 20))
            self.char_bi_lstm = dy.BiRNNBuilder(1, 20, 128, self.model, dy.LSTMBuilder)

        # Word LSTM parameters
        if use_char_rnn:
            input_dim = word_embedding_dim + 128
        else:
            input_dim = word_embedding_dim
        self.word_bi_lstm = dy.BiRNNBuilder(num_lstm_layers, input_dim, hidden_dim, self.model, dy.LSTMBuilder)

        # Matrix that maps from Bi-LSTM output to num tags
        self.lstm_to_tags_params = {}
        self.lstm_to_tags_bias = {}
        self.mlp_out = {}
        self.mlp_out_bias = {}
        for att, set_size in tagset_sizes.items():
            self.lstm_to_tags_params[att] = self.model.add_parameters((set_size, hidden_dim))
            self.lstm_to_tags_bias[att] = self.model.add_parameters(set_size)
            self.mlp_out[att] = self.model.add_parameters((set_size, set_size))
            self.mlp_out_bias[att] = self.model.add_parameters(set_size)


    def word_rep(self, word):
        '''
        :param word: index of word in lookup table
        '''
        if self.lowercase_words:
            lower_word_form = i2w[word].lower()
            if lower_word_form in w2i:
                word_in_ds = w2i[lower_word_form]
            else:
                word_in_ds = word
        else:
            word_in_ds = word
        wemb = self.words_lookup[word_in_ds]
        if self.use_char_rnn:
            pad_char = c2i[PADDING_CHAR]
            # Note: use original casing ("word") for characters
            char_ids = [pad_char] + [c2i[c] for c in i2w[word]] + [pad_char] # TODO optimize
            char_embs = [self.char_lookup[cid] for cid in char_ids]
            char_exprs = self.char_bi_lstm.transduce(char_embs)
            return dy.concatenate([ wemb, char_exprs[-1] ])
        else:
            return wemb


    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        embeddings = [self.word_rep(w) for w in sentence]

        lstm_out = self.word_bi_lstm.transduce(embeddings)

        H = {}
        Hb = {}
        O = {}
        Ob = {}
        scores = {}
        for att in self.attributes:        
            H[att] = dy.parameter(self.lstm_to_tags_params[att])
            Hb[att] = dy.parameter(self.lstm_to_tags_bias[att])
            O[att] = dy.parameter(self.mlp_out[att])
            Ob[att] = dy.parameter(self.mlp_out_bias[att])
            scores[att] = []
            for rep in lstm_out:
                score_t = O[att] * dy.tanh(H[att] * rep + Hb[att]) + Ob[att]
                scores[att].append(score_t)

        return scores


    def loss(self, sentence, tags_set):
        observations_set = self.build_tagging_graph(sentence)
        errors = {}
        for att, tags in tags_set.iteritems():
            err = []
            for obs, tag in zip(observations_set[att], tags):
                err_t = dy.pickneglogsoftmax(obs, tag)
                err.append(err_t)
            errors[att] = dy.esum(err)
        if self.att_props is not None:
            for att, err in errors.iteritems():
                prop_vec = dy.inputVector([self.att_props[att]] * err.dim()[0])
                err = dy.cmult(err, prop_vec)
        return errors


    def tag_sentence(self, sentence):
        observations_set = self.build_tagging_graph(sentence)
        tag_seqs = {}
        for att, observations in observations_set.iteritems():
            observations = [ dy.softmax(obs) for obs in observations ]
            probs = [ obs.npvalue() for obs in observations ]
            tag_seq = []
            for prob in probs:
                tag_t = np.argmax(prob)
                tag_seq.append(tag_t)
            tag_seqs[att] = tag_seq
        return tag_seqs

    
    def set_dropout(self, p):
        self.word_bi_lstm.set_dropout(p)


    def disable_dropout(self):
        self.word_bi_lstm.disable_dropout()

    def save(self, file_name):
        members_to_save = []
        members_to_save.append(self.words_lookup)
        if (self.use_char_rnn):
            members_to_save.append(self.char_lookup)
            members_to_save.append(self.char_bi_lstm)
        members_to_save.append(self.word_bi_lstm)
        members_to_save.extend(utils.sortvals(self.lstm_to_tags_params))
        members_to_save.extend(utils.sortvals(self.lstm_to_tags_bias))
        members_to_save.extend(utils.sortvals(self.mlp_out))
        members_to_save.extend(utils.sortvals(self.mlp_out_bias))
        self.model.save(file_name, members_to_save)
        
        with open(file_name + "-atts", 'w') as attdict:
            attdict.write("\t".join(sorted(self.attributes)))

    @property
    def model(self):
        return self.model


def get_att_prop(instances):
    logging.info("Calculating attribute proportions for proportional loss margin or proportional loss magnitude")
    total_tokens = 0
    att_counts = Counter()
    for instance in instances:
        total_tokens += len(instance.sentence)
        for att, tags in instance.tags.items():
            t2i = t2is[att]
            att_counts[att] += len([t for t in tags if t != t2i.get(NONE_TAG, -1)])
    return {att:(1.0 - (att_counts[att] / total_tokens)) for att in att_counts}

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--num-epochs", default=20, dest="num_epochs", type=int, help="Number of full passes through training set")
parser.add_argument("--lstm-layers", default=2, dest="lstm_layers", type=int, help="Number of LSTM layers")
parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="Size of LSTM hidden layers")
parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate")
parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="Amount of dropout to apply to LSTM part of graph")
parser.add_argument("--viterbi", dest="viterbi", action="store_true", help="Use viterbi training instead of CRF")
parser.add_argument("--loss-margin", default="one", dest="loss_margin", help="Loss margin calculation method in sequence tagger (currently only supported in Viterbi). Supported values - one (default), zero, att-prop (attribute proportional)")
parser.add_argument("--no-sequence-model", dest="no_sequence_model", action="store_true", help="Use regular LSTM tagger with no viterbi")
parser.add_argument("--loss-prop", dest="loss_prop", action="store_true", help="Proportional loss magnitudes in LSTM model")
parser.add_argument("--use-char-rnn", dest="use_char_rnn", action="store_true", help="Use character RNN")
parser.add_argument("--lowercase-words", dest="lowercase_words", action="store_true", help="Words are all in lowercased form (characters stay the same)")
parser.add_argument("--semi-supervised", dest="semi_supervised", action="store_true", help="Add KL-div term")
parser.add_argument("--kl-weight", default=1000, dest="kl_weight", type=float, help="Weight of KL-divergence term")
parser.add_argument("--log-dir", default="log", dest="log_dir", help="Directory where to write logs / serialized models")
parser.add_argument("--pos-separate-col", default=True, dest="pos_separate_col", help="Output examples have POS in separate column")
parser.add_argument("--dynet-mem", help="Ignore this outside argument")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
options = parser.parse_args()


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
if not os.path.exists(options.log_dir):
    os.mkdir(options.log_dir)
logging.basicConfig(filename=options.log_dir + "/log.txt", filemode="w", format="%(message)s", level=logging.INFO)
train_dev_cost = utils.CSVLogger(options.log_dir + "/train_dev.log", ["Train.cost", "Dev.cost"])


# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
if options.viterbi:
    objective = "Viterbi"
elif options.no_sequence_model:
    objective = "No Sequence Model"
else:
    objective = "CRF"
if options.viterbi:
    loss_scheme = "Viterbi margin scheme: {}".format(options.loss_margin)
elif options.no_sequence_model:
    loss_scheme = "LSTM loss weights proportional to attribute frequency: {}".format(options.loss_prop)
else:
    loss_scheme = "No loss variables in this objective"
logging.info(
"""
Dataset: {}
Pretrained Embeddings: {}
Num Epochs: {}
LSTM: {} layers, {} hidden dim
Initial Learning Rate: {}
Dropout: {}
Objective: {}
{}
Lowercasing words: {}

""".format(options.dataset, options.word_embeddings, options.num_epochs, options.lstm_layers, options.hidden_dim,
           options.learning_rate, options.dropout, objective, loss_scheme, options.lowercase_words))

if options.debug:
    print "DEBUG MODE"

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = cPickle.load(open(options.dataset, "r"))
w2i = dataset["w2i"]
t2is = dataset["t2is"]
c2i = dataset["c2i"]
m2i = None
i2w = { i: w for w, i in w2i.items() } # Inverse mapping
i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in t2is.items() }
i2c = { i: c for c, i in c2i.items() }

tag_lists = { att: [ i2t[idx] for idx in xrange(len(i2t)) ] for att, i2t in i2ts.items() } # To use in the confusion matrix
training_instances = dataset["training_instances"]
training_vocab = dataset["training_vocab"]
dev_instances = dataset["dev_instances"]
dev_vocab = dataset["dev_vocab"]
test_instances = dataset["test_instances"]


# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===
if options.word_embeddings is not None:
    word_embeddings = utils.read_pretrained_embeddings(options.word_embeddings, w2i)
else:
    word_embeddings = None

tag_set_sizes = { att: len(t2i) for att, t2i in t2is.items() }
if options.no_sequence_model:
    if options.loss_prop:
        att_props = get_att_prop(training_instances)
    else:
        att_props = None
    model = LSTMTagger(tagset_sizes=tag_set_sizes,
                       num_lstm_layers=options.lstm_layers,
                       hidden_dim=options.hidden_dim,
                       word_embeddings=word_embeddings,
                       train_vocab_ctr=training_vocab,
                       use_char_rnn=options.use_char_rnn,
                       charset_size=len(c2i),
                       lowercase_words=options.lowercase_words,
                       vocab_size=len(w2i),
                       att_props=att_props,
                       word_embedding_dim=DEFAULT_WORD_EMBEDDING_SIZE)

else:
    if not options.viterbi:
        margins = None
    elif options.loss_margin == "one":
        margins = {att:1.0 for att in t2is.keys()}
    elif options.loss_margin == "zero":
        margins = {att:0.0 for att in t2is.keys()}
    elif options.loss_margin == "att-prop":
        margins = get_att_prop(training_instances)
    model = BiLSTM_CRF(tag_set_sizes,
                       options.lstm_layers,
                       options.hidden_dim,
                       word_embeddings,
                       options.use_char_rnn,
                       len(c2i),
                       training_vocab,
                       margins,
                       options.lowercase_words,
                       vocab_size=len(w2i))

if options.semi_supervised: # save initial embeddings for KL term
    arr_look = model.words_lookup.as_array()
    if options.debug:
        print "Words lookup is a table of size: {}".format(model.words_lookup.shape())
        print "Words lookup table as array size: {} with lines of length: {}".format(len(arr_look), len(arr_look[200]))
    bar = progressbar.ProgressBar()
    init_embs = []
    for we in bar(arr_look):
        init_embs.append(list(we))
    if options.debug:
        print "Copied words lookup table size: {}, record size: {}".format(len(init_embs), len(init_embs[300]))
    embs_shape = model.words_lookup.shape()

trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9, 0.1)
logging.info("Training Algorithm: {}".format(type(trainer)))

logging.info("Number training instances: {}".format(len(training_instances)))
logging.info("Number dev instances: {}".format(len(dev_instances)))

for epoch in xrange(int(options.num_epochs)):
    bar = progressbar.ProgressBar()
    random.shuffle(training_instances)
    train_loss = 0.0
    train_correct = Counter()
    train_total = Counter()

    if options.dropout > 0:
        model.set_dropout(options.dropout)

    if options.debug:
        train_instances = training_instances[0:int(len(training_instances)/20)]
    else:
        train_instances = training_instances
    
    for idx,instance in enumerate(bar(train_instances)):
        if len(instance.sentence) == 0: continue

        # TODO make the interface all the same here
        if options.viterbi:
            losses = []
            gold_tags = instance.tags
            for att in model.attributes:
                if att not in instance.tags:
                    gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
            loss_exprs, viterbi_tags_set = model.viterbi_loss(instance.sentence, gold_tags)
            for att, tags in gold_tags.items():
                vit_tags = viterbi_tags_set[att]
                l = loss_exprs[att].scalar_value()
                # Record some info for training accuracy
                if l > 0:
                    for gold, viterbi in zip(tags, vit_tags):
                        if gold == viterbi:
                            train_correct[att] += 1
                else:
                    train_correct[att] += len(tags)
                train_total[att] += len(tags)
                losses.append(l)
            loss_expr = dy.esum(loss_exprs.values())
        elif options.no_sequence_model:
            gold_tags = instance.tags
            for att in model.attributes:
                if att not in instance.tags:
                    gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
            loss_exprs = model.loss(instance.sentence, gold_tags)
            loss_expr = dy.esum(loss_exprs.values())
        else:
            loss_exprs = model.neg_log_loss(instance.sentence, instance.tags)
            loss_expr = dy.esum(loss_exprs.values())
        if options.semi_supervised:# and idx > 100:
            frozen_embs = dy.nobackprop(dy.transpose(dy.concatenate_cols([ dy.inputVector(init_embs[i]) for i in instance.sentence ])))
            #frozen_embs = dy.transpose(dy.concatenate_cols([ dy.inputVector(init_embs[i]) for i in instance.sentence ]))
            embeddings_tensor = dy.transpose(dy.concatenate_cols([ model.words_lookup[i] for i in instance.sentence ]))
            #if options.debug and idx % 1000 == 999:
                #all_frozen_embs = dy.nobackprop(dy.transpose(dy.concatenate_cols([ dy.inputVector(init_embs[i]) for i in xrange(embs_shape[0]) ])))
                #all_embeddings_tensor = dy.transpose(dy.concatenate_cols([ model.words_lookup[i] for i in xrange(embs_shape[0]) ]))
                #print [all(all_frozen_embs.value()[i] == all_embeddings_tensor.value()[i]) for i in range(50,100,5)]
            kl_weight_expr = dy.inputVector([options.kl_weight])
            kl_div = utils.kl_div(embeddings_tensor, frozen_embs)
            weighted_kl_div = dy.cmult(kl_div, kl_weight_expr)
            if options.debug:
                print "KL Div {} with weight {} added to loss {}".format(kl_div.value(), options.kl_weight, loss_expr.value())
            loss_expr = loss_expr + weighted_kl_div
        loss = loss_expr.scalar_value()
        
        # Bail if loss is NaN
        if math.isnan(loss):
            assert False, "NaN occured"

        train_loss += (loss / len(instance.sentence))

        # Do backward pass and update parameters
        loss_expr.backward()
        trainer.update()

    logging.info("\n")
    logging.info("Epoch {} complete".format(epoch + 1))
    trainer.update_epoch(1)
    print trainer.status()

    # Evaluate dev data
    model.disable_dropout()
    dev_loss = 0.0
    dev_correct = Counter()
    dev_total = Counter()
    dev_oov_total = Counter()
    bar = progressbar.ProgressBar()
    total_wrong = Counter()
    total_wrong_oov = Counter()
    f1_eval = Evaluator(m = 'att')
    if options.debug:
        d_instances = dev_instances[0:int(len(dev_instances)/10)]
    else:
        d_instances = dev_instances
    with open("{}/devout_epoch-{:02d}.txt".format(options.log_dir, epoch + 1), 'w') as dev_writer:
        for instance in bar(d_instances):
            if len(instance.sentence) == 0: continue
            if options.no_sequence_model:
                gold_tags = instance.tags
                for att in model.attributes:
                    if att not in instance.tags:
                        gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
                losses = model.loss(instance.sentence, gold_tags)
                total_loss = sum([l.scalar_value() for l in losses.values()]) # TODO or average
                out_tags_set = model.tag_sentence(instance.sentence)
            else:
                gold_tags = instance.tags
                for att in model.attributes:
                    if att not in instance.tags:
                        gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
                losses = model.neg_log_loss(instance.sentence, gold_tags)
                total_loss = sum([l.value() for l in losses.values()]) # TODO or average
                _, out_tags_set = model.viterbi_loss(instance.sentence, gold_tags, use_margins=False)
                
            gold_strings = utils.morphotag_strings(i2ts, gold_tags, options.pos_separate_col)
            obs_strings = utils.morphotag_strings(i2ts, out_tags_set, options.pos_separate_col)
            dev_writer.write(("\n"
                             + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                         gold_strings, obs_strings)])
                             + "\n").encode('utf8'))
            for g, o in zip(gold_strings, obs_strings):
                f1_eval.add_instance(utils.split_tagstring(g, has_pos=True), utils.split_tagstring(o, has_pos=True))
            for att, tags in gold_tags.items():
                out_tags = out_tags_set[att]
                correct_sent = True

                for word, gold, out in zip(instance.sentence, tags, out_tags):
                    if gold == out:
                        dev_correct[att] += 1
                    else:
                        # Got the wrong tag
                        total_wrong[att] += 1
                        correct_sent = False
                        if i2w[word] not in training_vocab:
                            total_wrong_oov[att] += 1
                    
                    if i2w[word] not in training_vocab:
                        dev_oov_total[att] += 1
                # if not correct_sent:
                #     sent, tags = utils.convert_instance(instance, i2w, i2t)
                #     for i in range(len(sent)):
                #         logging.info( sent[i] + "\t" + tags[i] + "\t" + i2t[viterbi_tags[i]] )
                #     logging.info( "\n\n\n" )
                dev_total[att] += len(tags)
                
            dev_loss += (total_loss / len(instance.sentence))

    if options.viterbi:
        logging.info("POS Train Accuracy: {}".format(train_correct[POS_KEY] / train_total[POS_KEY]))
    logging.info("POS Dev Accuracy: {}".format(dev_correct[POS_KEY] / dev_total[POS_KEY]))
    logging.info("POS % OOV accuracy: {}".format((dev_oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / dev_oov_total[POS_KEY]))
    if total_wrong[POS_KEY] > 0:
        logging.info("POS % Wrong that are OOV: {}".format(total_wrong_oov[POS_KEY] / total_wrong[POS_KEY]))
    for attr in t2is.keys():
        if attr != POS_KEY:
            logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
    logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), not options.pos_separate_col))

    logging.info("Total dev tokens: {}, Total dev OOV: {}, % OOV: {}".format(dev_total[POS_KEY], dev_oov_total[POS_KEY], dev_oov_total[POS_KEY] / dev_total[POS_KEY]))
    
    train_loss = train_loss / len(train_instances)
    dev_loss = dev_loss / len(d_instances)
    logging.info("Train Loss: {}".format(train_loss))
    logging.info("Dev Loss: {}".format(dev_loss))
    train_dev_cost.add_column([train_loss, dev_loss])
    
    # Serialize model
    if not options.debug:
        new_model_file_name = "{}/model_epoch-{:02d}.bin".format(options.log_dir, epoch + 1)
        logging.info("Saving model to {}".format(new_model_file_name))
        model.save(new_model_file_name) # TODO also save non-internal model stuff like mappings
        if epoch > 1 and epoch % 10 != 0: # leave models from epochs 1,10,20, etc.
            logging.info("Removing files from previous epoch.")
            old_model_file_name = "{}/model_epoch-{:02d}.bin".format(options.log_dir, epoch)
            os.remove(old_model_file_name)
            os.remove(old_model_file_name + ".pym")
            os.remove(old_model_file_name + ".pyk")
            os.remove(old_model_file_name + "-atts")
            old_devout_file_name = "{}/devout_epoch-{:02d}.txt".format(options.log_dir, epoch)
            os.remove(old_devout_file_name)



# Evaluate test data (once)
logging.info("\n")
logging.info("Number test instances: {}".format(len(test_instances)))
model.disable_dropout()
test_correct = Counter()
test_total = Counter()
test_oov_total = Counter()
bar = progressbar.ProgressBar()
total_wrong = Counter()
total_wrong_oov = Counter()
f1_eval = Evaluator(m = 'att')
if options.debug:
    t_instances = test_instances[0:int(len(test_instances)/10)]
else:
    t_instances = test_instances
with open("{}/testout.txt".format(options.log_dir), 'w') as test_writer:
    for instance in bar(t_instances):
        if len(instance.sentence) == 0: continue
        if options.no_sequence_model:
            gold_tags = instance.tags
            for att in model.attributes:
                if att not in instance.tags:
                    gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
            out_tags_set = model.tag_sentence(instance.sentence)
        else:
            gold_tags = instance.tags
            for att in model.attributes:
                if att not in instance.tags:
                    gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
            _, out_tags_set = model.viterbi_loss(instance.sentence, gold_tags, use_margins=False)
            
        gold_strings = utils.morphotag_strings(i2ts, gold_tags, options.pos_separate_col)
        obs_strings = utils.morphotag_strings(i2ts, out_tags_set, options.pos_separate_col)
        test_writer.write(("\n"
                         + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                     gold_strings, obs_strings)])
                         + "\n").encode('utf8'))
        for g, o in zip(gold_strings, obs_strings):
            f1_eval.add_instance(utils.split_tagstring(g, has_pos=True), utils.split_tagstring(o, has_pos=True))
        for att, tags in gold_tags.items():
            out_tags = out_tags_set[att]
            correct_sent = True

            for word, gold, out in zip(instance.sentence, tags, out_tags):
                if gold == out:
                    test_correct[att] += 1
                else:
                    # Got the wrong tag
                    total_wrong[att] += 1
                    correct_sent = False
                    if i2w[word] not in training_vocab:
                        total_wrong_oov[att] += 1
                
                if i2w[word] not in training_vocab:
                    test_oov_total[att] += 1
            # if not correct_sent:
            #     sent, tags = utils.convert_instance(instance, i2w, i2t)
            #     for i in range(len(sent)):
            #         logging.info( sent[i] + "\t" + tags[i] + "\t" + i2t[viterbi_tags[i]] )
            #     logging.info( "\n\n\n" )
            test_total[att] += len(tags)

logging.info("POS Test Accuracy: {}".format(test_correct[POS_KEY] / test_total[POS_KEY]))
logging.info("POS % OOV accuracy: {}".format((test_oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / test_oov_total[POS_KEY]))
if total_wrong[POS_KEY] > 0:
    logging.info("POS % Wrong that are OOV: {}".format(total_wrong_oov[POS_KEY] / total_wrong[POS_KEY]))
for attr in t2is.keys():
    if attr != POS_KEY:
        logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), not options.pos_separate_col))

logging.info("Total test tokens: {}, Total test OOV: {}, % OOV: {}".format(test_total[POS_KEY], test_oov_total[POS_KEY], test_oov_total[POS_KEY] / test_total[POS_KEY]))
