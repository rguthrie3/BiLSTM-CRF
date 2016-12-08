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

    def __init__(self, tagset_size, num_lstm_layers, hidden_dim, word_embeddings, morpheme_embeddings, morpheme_projection, morpheme_decomps, train_vocab_ctr):
        self.model = dy.Model()
        self.tagset_size = tagset_size
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
        
        # Matrix that maps from Bi-LSTM output to num tags
        self.lstm_to_tags_params = self.model.add_parameters((tagset_size, hidden_dim))
        self.lstm_to_tags_bias = self.model.add_parameters(tagset_size)
        self.mlp_out = self.model.add_parameters((tagset_size, tagset_size))
        self.mlp_out_bias = self.model.add_parameters(tagset_size)

        # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
        self.transitions = self.model.add_lookup_parameters((tagset_size, tagset_size))


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


    def build_tagging_graph(self, sentence, dropout):
        dy.renew_cg()

        #embeddings = [self.word_rep(w) for w in sentence]
        embeddings = [self.words_lookup[w] for w in sentence]

        lstm_out = self.bi_lstm.transduce(embeddings)
        
        H = dy.parameter(self.lstm_to_tags_params)
        Hb = dy.parameter(self.lstm_to_tags_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)
        scores = []
        for rep in lstm_out:
            score_t = O * dy.tanh(H * rep + Hb) + Ob
            scores.append(score_t)

        return scores


    def score_sentence(self, observations, tags):
        assert len(observations) == len(tags)
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [t2i["<START>"]] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(self.transitions[tags[i+1]], tags[i]) + dy.pick(obs, tags[i+1])
            score_seq.append(score.value())
        score = score + dy.pick(self.transitions[t2i["<STOP>"]], tags[-1])
        return score


    def viterbi_loss(self, sentence, tags, dropout=False):
        observations = self.build_tagging_graph(sentence, dropout)
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations)
        if viterbi_tags != tags:
            gold_score = self.score_sentence(observations, tags)
            return (viterbi_score - gold_score), viterbi_tags
        else:
            return dy.scalarInput(0), viterbi_tags


    def neg_log_loss(self, sentence, tags, dropout=True):
        observations = self.build_tagging_graph(sentence, dropout)
        gold_score = self.score_sentence(observations, tags)
        forward_score = self.forward(observations)
        return forward_score - gold_score


    def forward(self, observations):

        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.tagset_size)
            return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))

        init_alphas = [-1e10] * self.tagset_size
        init_alphas[t2i["<START>"]] = 0
        for_expr = dy.inputVector(init_alphas)
        for i, obs in enumerate(observations):
            alphas_t = []
            for next_tag in range(self.tagset_size):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.tagset_size)
                next_tag_expr = for_expr + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.transitions[t2i["<STOP>"]]
        alpha = log_sum_exp(terminal_expr)
        return alpha


    def viterbi_decoding(self, observations):
        backpointers = []
        init_vvars   = [-1e10] * self.tagset_size
        init_vvars[t2i["<START>"]] = 0 # <Start> has all the probability
        for_expr     = dy.inputVector(init_vvars)
        trans_exprs  = [self.transitions[idx] for idx in range(self.tagset_size)]
        for i, obs in enumerate(observations):
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.tagset_size):
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


# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--word-embeddings", required=True, dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--morpheme-embeddings", dest="morpheme_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--morpheme-projection", dest="morpheme_projection", help="Pickle file containing projection matrix if applicable")
parser.add_argument("--num-epochs", default=50, dest="num_epochs", type=int, help="Number of full passes through training set")
parser.add_argument("--lstm-layers", default=2, dest="lstm_layers", type=int, help="Number of LSTM layers")
parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="Size of LSTM hidden layers")
parser.add_argument("--learning-rate", default=0.001, dest="learning_rate", type=float, help="Initial learning rate")
parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="Amount of dropout to apply to LSTM part of graph")
parser.add_argument("--viterbi", dest="viterbi", action="store_true", help="Use viterbi training instead of CRF")
parser.add_argument("--log-dir", default="log", dest="log_dir", help="Directory where to write logs / serialized models")
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


# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = cPickle.load(open(options.dataset, "r"))
w2i = dataset["w2i"]
w2i["<UNK>"] = len(w2i) # read the comment in word_rep to see why this is necessary
t2i = dataset["t2i"]
#m2i = dataset["m2i"]
m2i = None
t2i["<START>"] = len(t2i)
t2i["<STOP>"] = len(t2i)
i2w = { i: w for w, i in w2i.items() } # Inverse mapping
i2t = { i: t for t, i in t2i.items() }
tag_list = [ i2t[idx] for idx in xrange(len(i2t)) ] # To use in the confusion matrix
training_instances = dataset["training_instances"]
training_vocab = dataset["training_vocab"]
dev_instances = dataset["dev_instances"]
dev_vocab = dataset["dev_vocab"]


# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===
word_embeddings = utils.read_pretrained_embeddings(options.word_embeddings, w2i)
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
bilstm_crf = BiLSTM_CRF(len(t2i), options.lstm_layers, options.hidden_dim, word_embeddings, morpheme_embeddings, morpheme_projection, morpheme_decomps, training_vocab)
trainer = dy.AdamTrainer(bilstm_crf.model, options.learning_rate)

print "Number training instances:", len(training_instances)
print "Number Dev instances:", len(dev_instances)

for epoch in xrange(int(options.num_epochs)):
    bar = progressbar.ProgressBar()
    random.shuffle(training_instances)
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    if options.dropout > 0:
        bilstm_crf.set_dropout(options.dropout)
    for instance in bar(training_instances):
        if len(instance.sentence) == 0: continue
        if options.viterbi:
            loss_expr, viterbi_tags = bilstm_crf.viterbi_loss(instance.sentence, instance.tags)
            loss = loss_expr.scalar_value()
            # Record some info for training accuracy
            if loss > 0:
                for gold, viterbi in zip(instance.tags, viterbi_tags):
                    if gold == viterbi:
                        train_correct += 1
            else:
                train_correct += len(instance.tags)
            train_total += len(instance.tags)
        else:
            loss_expr = bilstm_crf.neg_log_loss(instance.sentence, instance.tags)
            loss = loss_expr.scalar_value()

        # Bail if loss is NaN
        if math.isnan(loss):
            sent, tags = utils.convert_instance(instance, i2w, i2t)
            print sent
            print tags
            for word in instance.sentence:
                print i2w[word], bilstm_crf.words_lookup[word].npvalue()
                print "\n\n"
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
    bilstm_crf.disable_dropout()
    dev_loss = 0.0
    dev_correct = 0
    dev_total = 0
    bar = progressbar.ProgressBar()
    total_wrong = 0
    total_wrong_oov = 0
    for instance in bar(dev_instances):
        loss = bilstm_crf.neg_log_loss(instance.sentence, instance.tags, dropout=False)
        dev_loss += (loss.value() / len(instance.sentence))
        viterbi_loss, viterbi_tags = bilstm_crf.viterbi_loss(instance.sentence, instance.tags)
        correct_sent = True
        for i, (gold, viterbi) in enumerate(zip(instance.tags, viterbi_tags)):
            if gold == viterbi:
                dev_correct += 1
            else:
                # Got the wrong tag
                total_wrong += 1
                correct_sent = False
                if i2w[instance.sentence[i]] not in training_vocab:
                    total_wrong_oov += 1
        # if not correct_sent:
        #     sent, tags = utils.convert_instance(instance, i2w, i2t)
        #     logging.info(sent)
        #     logging.info(tags)
        #     logging.info( [i2t[t] for t in viterbi_tags] )
        #     logging.info( "\n\n\n" )
                
        dev_total += len(instance.tags)
    if options.viterbi:
        logging.info("Train Accuracy: {}".format(float(train_correct) / train_total))
    logging.info("Dev Accuracy: {}".format(float(dev_correct) / dev_total))
    if total_wrong > 0:
        logging.info("% Wrong that are OOV: {}".format(float(total_wrong_oov) / total_wrong))

    train_loss = train_loss / len(training_instances)
    dev_loss = dev_loss / len(dev_instances)
    logging.info("Train Loss: {}".format(train_loss))
    logging.info("Dev Loss: {}".format(dev_loss))
    train_dev_cost.add_column([train_loss, dev_loss])
