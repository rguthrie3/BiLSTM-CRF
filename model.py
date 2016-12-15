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


class BiLSTM_CRF:

    def __init__(self, tagset_sizes, num_lstm_layers, hidden_dim, word_embeddings, morpheme_embeddings, morpheme_projection, morpheme_decomps, train_vocab_ctr):
        self.model = dy.Model()
        self.tagset_sizes = tagset_sizes
        self.train_vocab_ctr = train_vocab_ctr

        # Word embedding parameters
        vocab_size = word_embeddings.shape[0]
        word_embedding_dim = word_embeddings.shape[1]
        self.words_lookup = self.model.add_lookup_parameters((vocab_size, word_embedding_dim))
        self.words_lookup.init_from_array(word_embeddings)

        # Morpheme embedding parameters
        # morpheme_vocab_size = morpheme_embeddings.shape[0]
        # morpheme_embedding_dim = morpheme_embeddings.shape[1]
        # self.morpheme_lookup = self.model.add_lookup_parameters((morpheme_vocab_size, morpheme_embedding_dim))
        # self.morpheme_lookup.init_from_array(morpheme_embeddings)
        # self.morpheme_decomps = morpheme_decomps

        # if morpheme_projection is not None:
        #     self.morpheme_projection = self.model.add_parameters((word_embedding_dim, morpheme_embedding_dim))
        #     self.morpheme_projection.init_from_array(morpheme_projection)
        # else:
        #     self.morpheme_projection = None

        # LSTM parameters
        self.bi_lstm = dy.BiRNNBuilder(num_lstm_layers, word_embedding_dim, hidden_dim, self.model, dy.LSTMBuilder)
        
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
        """
        For rare words in the training data, we will use their morphemes
        to make their representation
        """ 
        if self.train_vocab_ctr[word] > 5:
            return self.words_lookup[word]
        else:
            # Use morpheme embeddings
            morpheme_decomp = self.morpheme_decomps[word]
            rep = self.morpheme_lookup[morpheme_decomp[0]]
            for m in morpheme_decomp[1:]:
                rep += self.morpheme_lookup[m]
            if self.morpheme_projection is not None:
                rep = self.morpheme_projection * rep
            if np.linalg.norm(rep.npvalue()) >= 50.0:
                # This is meant to handle things like URLs and weird tokens like !!!!!!!!!!!!!!!!!!!!!
                # that are splitting into a lot of morphemes, and their large representations are cause NaNs
                # TODO handle this in a better way.  Looks like all such inputs are either URLs, email addresses, or
                # long strings of a punctuation token when the decomposition is > 10
                return self.words_lookup[w2i["<UNK>"]]
            return rep


    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        #embeddings = [self.word_rep(w) for w in sentence]
        embeddings = [self.words_lookup[w] for w in sentence]

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
            tags = [t2i["<NONE>"]] * len(observations)
        assert len(observations) == len(tags)
        trans = self.transitions[att]
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [t2i["<START>"]] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(trans[tags[i+1]], tags[i]) + dy.pick(obs, tags[i+1])
            score_seq.append(score.value())
        score = score + dy.pick(trans[t2i["<STOP>"]], tags[-1])
        return score


    def viterbi_loss(self, sentence, tags_set):
        observations_set = self.build_tagging_graph(sentence)
        losses = {}
        ret_tags = {}
        for att, observations in observations_set.items():
            tags = tags_set[att]
            viterbi_tags, viterbi_score = self.viterbi_decoding(observations, att)
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
        init_alphas[t2i["<START>"]] = 0
        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(tagset_size):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * tagset_size)
                next_tag_expr = for_expr + trans[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr, tagset_size))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + trans[t2i["<STOP>"]]
        alpha = log_sum_exp(terminal_expr, tagset_size)
        return alpha


    def viterbi_decoding(self, observations, att):
        t2i = t2is[att]
        tagset_size = self.tagset_sizes[att]
        backpointers = []
        init_vvars   = [-1e10] * tagset_size
        init_vvars[t2i["<START>"]] = 0 # <Start> has all the probability
        for_expr     = dy.inputVector(init_vvars)
        trans_exprs  = [self.transitions[att][idx] for idx in range(tagset_size)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(tagset_size):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id  = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[t2i["<STOP>"]]
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
        assert start == t2i["<START>"]
        # Return best path and best path's score
        return best_path, path_score

    @property
    def model(self):
        return self.model


# TODO support multi-attribute tagging
class LSTMTagger:

    def __init__(self, tagset_size, num_lstm_layers, hidden_dim, word_embeddings, train_vocab_ctr, use_char_rnn, charset_size, vocab_size=None, word_embedding_dim=None):
        self.model = dy.Model()
        self.tagset_size = tagset_size
        self.train_vocab_ctr = train_vocab_ctr

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
        self.lstm_to_tags_params = self.model.add_parameters((tagset_size, hidden_dim))
        self.lstm_to_tags_bias = self.model.add_parameters(tagset_size)
        self.mlp_out = self.model.add_parameters((tagset_size, tagset_size))
        self.mlp_out_bias = self.model.add_parameters(tagset_size)


    def word_rep(self, w):
        wemb = self.words_lookup[w]
        if self.use_char_rnn:
            pad_char = c2i["<*>"]
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

        H = dy.parameter(self.lstm_to_tags_params)
        Hb = dy.parameter(self.lstm_to_tags_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)
        scores = []
        for rep in lstm_out:
            score_t = O * dy.tanh(H * rep + Hb) + Ob
            scores.append(score_t)

        return scores


    def loss(self, sentence, tags_set):
        tags = tags_set["POS"]
        observations = self.build_tagging_graph(sentence)
        errors = []
        for obs, tag in zip(observations, tags):
            err_t = dy.pickneglogsoftmax(obs, tag)
            errors.append(err_t)
        return dy.esum(errors)


    def tag_sentence(self, sentence):
        observations = self.build_tagging_graph(sentence)
        observations = [ dy.softmax(obs) for obs in observations ]
        probs = [ obs.npvalue() for obs in observations ]
        tag_seq = []
        for prob in probs:
            tag_t = np.argmax(prob)
            tag_seq.append(tag_t)
        return tag_seq

    
    def set_dropout(self, p):
        self.word_bi_lstm.set_dropout(p)


    def disable_dropout(self):
        self.word_bi_lstm.disable_dropout()


# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--morpheme-embeddings", dest="morpheme_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--morpheme-projection", dest="morpheme_projection", help="Pickle file containing projection matrix if applicable")
parser.add_argument("--num-epochs", default=20, dest="num_epochs", type=int, help="Number of full passes through training set")
parser.add_argument("--lstm-layers", default=2, dest="lstm_layers", type=int, help="Number of LSTM layers")
parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="Size of LSTM hidden layers")
parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate")
parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="Amount of dropout to apply to LSTM part of graph")
parser.add_argument("--viterbi", dest="viterbi", action="store_true", help="Use viterbi training instead of CRF")
parser.add_argument("--no-sequence-model", dest="no_sequence_model", action="store_true", help="Use regular LSTM tagger with no viterbi")
parser.add_argument("--use-char-rnn", dest="use_char_rnn", action="store_true", help="Use character RNN")
parser.add_argument("--log-dir", default="log", dest="log_dir", help="Directory where to write logs / serialized models")
parser.add_argument("--dev-output", default="dev-out", dest="dev_output", help="File with output examples")
options = parser.parse_args()


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
if not os.path.exists(options.log_dir):
    os.mkdir(options.log_dir)
logging.basicConfig(filename=options.log_dir + "/log.txt", filemode="w", format="%(message)s", level=logging.INFO)
train_dev_cost = utils.CSVLogger(options.log_dir + "/train_dev.log", ["Train.cost", "Dev.cost"])
dev_writer = open(options.dev_output, 'w')


# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logging.info(
"""
Dataset: {}
Pretrained Embeddings: {}
Num Epochs: {}
LSTM: {} layers, {} hidden dim
Initial Learning Rate: {}
Dropout: {}
Objective: {}

""".format(options.dataset, options.word_embeddings, options.num_epochs, options.lstm_layers, options.hidden_dim,
           options.learning_rate, options.dropout, "Viterbi" if options.viterbi else "CRF"))


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
training_instances = dataset["training_instances"]
training_vocab = dataset["training_vocab"]
dev_instances = dataset["dev_instances"]
dev_vocab = dataset["dev_vocab"]


# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===
if options.word_embeddings is not None:
    word_embeddings = utils.read_pretrained_embeddings(options.word_embeddings, w2i)
else:
    word_embeddings = None


if options.no_sequence_model:
    model = LSTMTagger(tagset_size=len(t2is["POS"]),
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
    model = BiLSTM_CRF({ att: len(t2i) for att, t2i in t2is.items() },
                       options.lstm_layers,
                       options.hidden_dim,
                       word_embeddings,
                       morpheme_embeddings,
                       morpheme_projection,
                       morpheme_decomps,
                       training_vocab)


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

    for instance in bar(training_instances):
        if len(instance.sentence) == 0: continue

        # TODO make the interface all the same here
        if options.viterbi:
            loss = 0
            gold_tags = instance.tags
            for att in model.attributes:
                if att not in instance.tags:
                    gold_tags[att] = [t2is[att]["<NONE>"]] * len(instance.sentence)
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
                loss += l
            # TODO this is average. parametrize and allow other formulations
            loss /= len(gold_tags)
            loss_expr = dy.average(loss_exprs.values())
        elif options.no_sequence_model:
            loss_expr = model.loss(instance.sentence, instance.tags)
            loss = loss_expr.scalar_value()
        else:
            loss_exprs = model.neg_log_loss(instance.sentence, instance.tags)
            # TODO parametrize and allow other formulations
            loss_expr = dy.average(loss_exprs.values())
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
    dev_writer.write("\nepoch " + str(epoch) + "\n")
    for instance in bar(dev_instances):
        if len(instance.sentence) == 0: continue
        if options.no_sequence_model:
            loss = model.loss(instance.sentence, instance.tags)
            dev_loss += (loss.scalar_value() / len(instance.sentence))
            out_tags = model.tag_sentence(instance.sentence)
        else:
            gold_tags = instance.tags
            for att in model.attributes:
                if att not in instance.tags:
                    gold_tags[att] = [t2is[att]["<NONE>"]] * len(instance.sentence)
            losses = model.neg_log_loss(instance.sentence, gold_tags)
            total_loss = sum([l.value() for l in losses]) / len(instance.sentence)
            _, out_tags_set = model.viterbi_loss(instance.sentence, gold_tags)
            dl = 0.0
            dev_writer.write("\n"
                             +"\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                         ["|".join(att + "=" + i2ts[att][v] for att,vals in gold_tags.items()) for v in vals],
                                                                         ["|".join(att + "=" + i2ts[att][v] for att,vals in out_tags_set.items()) for v in vals])])
                             + "\n")
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
                
            dev_loss += (total_loss / len(gold_tags))

    if options.viterbi:
        for att in t2is.keys():
            logging.info("{} Train Accuracy: {}".format(att, train_correct[att] / train_total[att]))
    for att in t2is.keys():
        logging.info("{} Dev Accuracy: {}".format(att, dev_correct[att] / dev_total[att]))
        logging.info("{} % OOV accuracy: {}".format(att, (dev_oov_total[att] - total_wrong_oov[att]) / dev_oov_total[att]))
        if total_wrong[att] > 0:
            logging.info("{} % Wrong that are OOV: {}".format(att, total_wrong_oov[att] / total_wrong[att]))

    train_loss = train_loss / len(training_instances)
    dev_loss = dev_loss / len(dev_instances)
    logging.info("Train Loss: {}".format(train_loss))
    logging.info("Dev Loss: {}".format(dev_loss))
    train_dev_cost.add_column([train_loss, dev_loss])
