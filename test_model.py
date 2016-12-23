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


def get_next_att_batch(attributes, att_tuple, idx):
    ret = {}
    for i, att in enumerate(attributes, idx):
        ret[att] = att_tuple[i]
    return ret
    
class BiLSTM_CRF:

    def __init__(self, rnn_model, use_char_rnn, tagset_sizes, train_vocab_ctr, margins):
        self.tagset_sizes = tagset_sizes
        self.train_vocab_ctr = train_vocab_ctr
        self.margins = margins
        self.use_char_rnn = use_char_rnn
        
        self.model = dy.Model()
        att_tuple = self.model.load(rnn_model)
        self.attributes = open(rnn_model + "-atts", "r").read().split("\t") # can also be extracted then sorted from tagset_sizes
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
        """
        For rare words in the training data, we will use their morphemes
        to make their representation
        """ 
        if self.train_vocab_ctr[word] > 5 or self.morpheme_lookup is None:
            wemb = self.words_lookup[word]
        else:
            # Use morpheme embeddings
            morpheme_decomp = self.morpheme_decomps[word]
            wemb = self.morpheme_lookup[morpheme_decomp[0]]
            for m in morpheme_decomp[1:]:
                wemb += self.morpheme_lookup[m]
            if self.morpheme_projection is not None:
                wemb = self.morpheme_projection * wemb
            if np.linalg.norm(wemb.npvalue()) >= 50.0:
                # This is meant to handle things like URLs and weird tokens like !!!!!!!!!!!!!!!!!!!!!
                # that are splitting into a lot of morphemes, and their large representations are cause NaNs
                # TODO handle this in a better way.  Looks like all such inputs are either URLs, email addresses, or
                # long strings of a punctuation token when the decomposition is > 10
                wemb = self.words_lookup[w2i["<UNK>"]]
        
        if self.use_char_rnn:
            pad_char = c2i[PADDING_CHAR]
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


    def viterbi_loss(self, sentence, tags_set):
        observations_set = self.build_tagging_graph(sentence)
        losses = {}
        ret_tags = {}
        for att, observations in observations_set.items():
            tags = tags_set[att]
            viterbi_tags, viterbi_score = self.viterbi_decoding(observations, tags, att)
            if viterbi_tags != tags:
                gold_score = self.score_sentence(observations, tags, att)
                losses[att] = viterbi_score - gold_score
                ret_tags[att] = viterbi_tags
            else:
                losses[att] = dy.scalarInput(0)
                ret_tags[att] = viterbi_tags
        return losses, ret_tags


    def viterbi_decoding(self, observations, gold_tags, att):
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
            if self.margins[att] != 0:
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

    @property
    def model(self):
        return self.model

# LOADING NOT YET IMPLEMENTED
class LSTMTagger:

    def __init__(self, tagset_sizes, num_lstm_layers, hidden_dim, word_embeddings, train_vocab_ctr, use_char_rnn, charset_size, vocab_size=None, word_embedding_dim=None):
        self.model = dy.Model()
        self.tagset_sizes = tagset_sizes
        self.train_vocab_ctr = train_vocab_ctr
        self.attributes = tagset_sizes.keys()
        
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


    def word_rep(self, w):
        wemb = self.words_lookup[w]
        if self.use_char_rnn:
            pad_char = c2i[PADDING_CHAR]
            char_ids = [pad_char] + [c2i[c] for c in i2w[w]] + [pad_char] # TODO optimize
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
        for att, tags in tags_set.items():
            err = []
            for obs, tag in zip(observations_set[att], tags):
                err_t = dy.pickneglogsoftmax(obs, tag)
                err.append(err_t)
            errors[att] = dy.esum(err)
        return errors


    def tag_sentence(self, sentence):
        observations_set = self.build_tagging_graph(sentence)
        tag_seqs = {}
        for att, observations in observations_set.items():
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

    @property
    def model(self):
        return self.model


def get_att_prop(instances):
    logging.info("Calculating attribute proportions for proportional loss margin")
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
parser.add_argument("--model", required=True, dest="model_file", help="Model file to use (.bin)")
parser.add_argument("--use-char-rnn", dest="use_char_rnn", action="store_true", help="Model being read has char RNN trained")
parser.add_argument("--morpheme-projection", dest="morpheme_projection", help="Pickle file containing projection matrix if applicable")
parser.add_argument("--viterbi", dest="viterbi", action="store_true", help="Use viterbi training instead of CRF")
parser.add_argument("--loss-margin", default="one", dest="loss_margin", help="Loss margin calculation method in sequence tagger (currently only supported in Viterbi). Supported values - one (default), zero, att-prop (attribute proportional)")
parser.add_argument("--no-sequence-model", dest="no_sequence_model", action="store_true", help="Use regular LSTM tagger with no viterbi")
parser.add_argument("--out-dir", default="out", dest="out_dir", help="Directory where to write output")
parser.add_argument("--pos-separate-col", default=True, dest="pos_separate_col", help="Output examples have POS in separate column")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
options = parser.parse_args()


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
if not os.path.exists(options.out_dir):
    os.mkdir(options.out_dir)
logging.basicConfig(filename=options.out_dir + "/out.txt", filemode="w", format="%(message)s", level=logging.INFO)


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
Model input: {}
Objective: {}
Viterbi margin scheme: {}

""".format(options.dataset, options.model_file, objective, options.loss_margin))

if options.debug:
    print "DEBUG MODE"

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = cPickle.load(open(options.dataset, "r"))
w2i = dataset["w2i"]
t2is = dataset["t2is"]
c2i = dataset["c2i"]
#m2i = dataset["m2i"]
m2i = None
i2w = { i: w for w, i in w2i.items() } # Inverse mapping
i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in t2is.items() }
i2c = { i: c for c, i in c2i.items() }

tag_lists = { att: [ i2t[idx] for idx in xrange(len(i2t)) ] for att, i2t in i2ts.items() } # To use in the confusion matrix
training_vocab = dataset["training_vocab"]
test_instances = dataset["test_instances"]


# ===-----------------------------------------------------------------------===
# Load model
# ===-----------------------------------------------------------------------===

tag_set_sizes = { att: len(t2i) for att, t2i in t2is.items() }
if options.no_sequence_model: # NOT IMPLEMENTED YET
    model = LSTMTagger(tagset_sizes=tag_set_sizes,
                       num_lstm_layers=options.lstm_layers,
                       hidden_dim=options.hidden_dim,
                       word_embeddings=word_embeddings,
                       train_vocab_ctr=training_vocab,
                       use_char_rnn=options.use_char_rnn,
                       charset_size=len(c2i),
                       vocab_size=len(w2i),
                       word_embedding_dim=128)

else:
    #morpheme_embeddings = utils.read_pretrained_embeddings(options.morpheme_embeddings, m2i)
    # if options.morpheme_projection is not None:
    #     assert word_embeddings.shape[1] != morpheme_embeddings.shape[1]
    #     morpheme_projection = cPickle.load(open(options.morpheme_projection, "r"))
    # else:
    #     morpheme_projection = None

    morpheme_embeddings = None
    morpheme_projection = None
    morpheme_decomps = None
    #morpheme_decomps = dataset["morpheme_segmentations"]
    if not options.viterbi:
        margins = None
    elif options.loss_margin == "one":
        margins = {att:1.0 for att in t2is.keys()}
    elif options.loss_margin == "zero":
        margins = {att:0.0 for att in t2is.keys()}
    elif options.loss_margin == "att-prop":
        margins = get_att_prop(dataset["training_instances"])
    model = BiLSTM_CRF(options.model_file, options.use_char_rnn, tag_set_sizes, training_vocab, margins)

logging.info("Number test instances: {}".format(len(test_instances)))

# Evaluate test data
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
with open("{}/testout.txt".format(options.out_dir), 'w') as test_writer:
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
            _, out_tags_set = model.viterbi_loss(instance.sentence, gold_tags)
            
        gold_strings = utils.morphotag_strings(i2ts, gold_tags, options.pos_separate_col)
        obs_strings = utils.morphotag_strings(i2ts, out_tags_set, options.pos_separate_col)
        test_writer.write("\n"
                         + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                     gold_strings, obs_strings)])
                         + "\n")
        for g, o in zip(gold_strings, obs_strings):
            f1_eval.add_instance(utils.split_tagstring(g), utils.split_tagstring(o))
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
    