from __future__ import division
from collections import Counter
from evaluate_morphotags import Evaluator
from sys import maxint

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

Instance = collections.namedtuple("Instance", ["word", "tags"])

NONE_TAG = "<NONE>"
POS_KEY = "POS"
PADDING_CHAR = "<*>"

DEFAULT_WORD_EMBEDDING_SIZE = 64
DEFAULT_HIDDEN_DIM = 128
DEFAULT_CHAR_HIDDEN_DIM = 50

FLOAT_FORMAT = "{:.4f}"

class MLP:

    def __init__(self, tagset_sizes, hidden_dim, char_hidden_dim, word_embeddings, we_update, use_char_rnn, charset_size, lowercase_words, att_props=None, vocab_size=None, word_embedding_dim=None):
        self.model = dy.Model()
        self.tagset_sizes = tagset_sizes
        self.attributes = tagset_sizes.keys()
        self.we_update = we_update
        self.lowercase_words = lowercase_words
        if att_props is not None:
            self.att_props = {att:(1.0-p) for att,p in att_props.iteritems()}
        else:
            self.att_props = None
        
        if word_embeddings is not None: # Use pretrained embeddings
            vocab_size = word_embeddings.shape[0]
            word_embedding_dim = word_embeddings.shape[1]
        
        self.words_lookup = self.model.add_lookup_parameters((vocab_size, word_embedding_dim))
        
        if word_embeddings is not None:
            self.words_lookup.init_from_array(word_embeddings)

        # Char LSTM Parameters
        self.use_char_rnn = use_char_rnn
        if use_char_rnn:
            self.char_lookup = self.model.add_lookup_parameters((charset_size, 20))
            self.char_bi_lstm = dy.BiRNNBuilder(1, 20, char_hidden_dim, self.model, dy.LSTMBuilder)

        # Word parameters
        if use_char_rnn:
            input_dim = word_embedding_dim + char_hidden_dim
        else:
            input_dim = word_embedding_dim

        # Matrix that maps from embeddings to num tags
        self.emb_to_hidden_params = self.model.add_parameters((hidden_dim, input_dim))
        self.emb_to_hidden_bias = self.model.add_parameters(hidden_dim)
        self.hidden_to_atts_params = {}
        self.hidden_to_atts_bias = {}
        for att, set_size in tagset_sizes.items():
            self.hidden_to_atts_params[att] = self.model.add_parameters((set_size, hidden_dim))
            self.hidden_to_atts_bias[att] = self.model.add_parameters(set_size)


    def word_rep(self, word):
        '''
        :param word: word in lookup table (not index)
        '''
        if self.lowercase_words:
            lower_word_form = word.lower()
            if lower_word_form in w2i:
                word_in_ds = w2i[lower_word_form]
            else:
                word_in_ds = w2i[word]
        else:
            word_in_ds = w2i[word]
        wemb = dy.lookup(self.words_lookup, word_in_ds, update=self.we_update)
        if self.use_char_rnn:
            pad_char = c2i[PADDING_CHAR]
            # Note: use original casing ("word") for characters
            char_ids = [pad_char] + [c2i[c] for c in word] + [pad_char] # TODO optimize
            char_embs = [self.char_lookup[cid] for cid in char_ids]
            char_exprs = self.char_bi_lstm.transduce(char_embs)
            return dy.concatenate([ wemb, char_exprs[-1] ])
        else:
            return wemb


    def build_tagging_graph(self, word):
        dy.renew_cg()

        embedding = self.word_rep(word)

        O = {}
        Ob = {}
        scores = {}
        
        # hidden layer computation
        H = dy.parameter(self.emb_to_hidden_params)
        Hb = dy.parameter(self.emb_to_hidden_bias)
        hidden_res = dy.tanh(H * embedding + Hb)
        
        # per-attribute computation
        for att in self.attributes:
            O[att] = dy.parameter(self.hidden_to_atts_params[att])
            Ob[att] = dy.parameter(self.hidden_to_atts_bias[att])
            scores[att] = O[att] * hidden_res + Ob[att]

        return scores


    def loss(self, word, tags):
        obss = self.build_tagging_graph(word)
        errors = {}
        for att, vals in tags.iteritems():
            if att == POS_KEY: continue
            if len(vals) == 0:
                print word, att
                exit()
            err = dy.esum([dy.pickneglogsoftmax(obss[att], v) for v in vals])
            if self.att_props is not None:
                prop_vec = dy.inputVector([self.att_props[att]])
                err = err * prop_vec
            errors[att] = err
        return errors


    def tag_word(self, word):
        # TODO: allow multiple vals per attribute (threshold? All above NONE?)
        obss = self.build_tagging_graph(word)
        tags = {}
        for att, obs in obss.iteritems():
            if att == POS_KEY: continue
            probs = dy.softmax(obs).npvalue() # softmax probably not needed here
            tags[att] = [np.argmax(probs)]
        return tags

    def save(self, file_name):
        members_to_save = []
        members_to_save.append(self.words_lookup)
        if (self.use_char_rnn):
            members_to_save.append(self.char_lookup)
            members_to_save.append(self.char_bi_lstm)           
        members_to_save.append(self.emb_to_hidden_params)
        members_to_save.append(self.emb_to_hidden_bias)
        members_to_save.extend(utils.sortvals(self.hidden_to_atts_params))
        members_to_save.extend(utils.sortvals(self.hidden_to_atts_bias))
        self.model.save(file_name, members_to_save)
        
        with open(file_name + "-atts", 'w') as attdict:
            attdict.write("\t".join(sorted(self.attributes)))

    @property
    def model(self):
        return self.model


def get_att_prop(instances):
    logging.info("Calculating attribute proportions for proportional loss margin or proportional loss magnitude")
    att_counts = Counter()
    for instance in instances:
        for att in instance.tags:
            att_counts[att] += 1
    total_tokens = len(instances)
    return {att:(1.0 - (att_counts[att] / total_tokens)) for att in att_counts}

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--num-epochs", default=100, dest="num_epochs", type=int, help="Number of full passes through training set")
parser.add_argument("--hidden-dim", default=DEFAULT_HIDDEN_DIM, dest="hidden_dim", type=int, help="Size of hidden layer")
parser.add_argument("--char-hidden-dim", default=DEFAULT_CHAR_HIDDEN_DIM, dest="char_hidden_dim", type=int, help="Size of hidden layer")
parser.add_argument("--training-size", default=maxint, dest="training_size", type=int, help="Max size of training set")
parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate")
parser.add_argument("--we-update", dest="we_update", action="store_true", help="Word Embeddings are updated")
parser.add_argument("--loss-prop", dest="loss_prop", action="store_true", help="Proportional loss magnitudes in MLP model")
parser.add_argument("--use-char-rnn", dest="use_char_rnn", action="store_true", help="Use character Bi-LSTM")
parser.add_argument("--lowercase-words", dest="lowercase_words", action="store_true", help="Words are all in lowercased form (characters stay the same)")
parser.add_argument("--log-dir", default="log", dest="log_dir", help="Directory where to write logs / serialized models")
parser.add_argument("--dynet-mem", help="Ignore this outside argument")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
options = parser.parse_args()


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
if not os.path.exists(options.log_dir):
    os.mkdir(options.log_dir)
logging.basicConfig(filename=options.log_dir + "/log.txt", filemode="w", format="%(message)s", level=logging.INFO)

# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
char_rnn_dim_str = "{} hidden dim".format(options.char_hidden_dim)\
                if options.use_char_rnn else "Not used"

logging.info(
"""
Dataset: {}
Pretrained Embeddings: {}
Num Epochs: {}
MLP: {} hidden dim
Char-LSTM: {}
Training set size limit: {}
Initial Learning Rate: {}
Lowercasing words: {}
""".format(options.dataset, options.word_embeddings, options.num_epochs, options.hidden_dim,\
                char_rnn_dim_str, options.training_size, options.learning_rate,\
                options.lowercase_words))

if options.debug:
    print "DEBUG MODE"

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = cPickle.load(open(options.dataset, "r"))
t2is = dataset["t2is"]
c2i = dataset["c2i"]
i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in t2is.items() } # Inverse mapping
i2c = { i: c for c, i in c2i.items() }

tag_lists = { att: [ i2t[idx] for idx in xrange(len(i2t)) ] for att, i2t in i2ts.items() } # To use in the confusion matrix
training_instances = dataset["training_instances"]
dev_instances = dataset["dev_instances"]
test_instances = dataset["test_instances"]

w2i = {}
w2i.update({ w.word: i for i,w in enumerate(training_instances) })
w2i.update({ w.word: i for i,w in enumerate(dev_instances) })
w2i.update({ w.word: i for i,w in enumerate(test_instances) })

# trim training set for size evaluation
if len(training_instances) > options.training_size:
    random.shuffle(training_instances)
    training_instances = training_instances[:options.training_size]

# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===
if options.word_embeddings is not None:
    word_embeddings = utils.read_pretrained_embeddings(options.word_embeddings, w2i)
else:
    word_embeddings = None

tag_set_sizes = { att: len(t2i) for att, t2i in t2is.items() }
if options.loss_prop:
    att_props = get_att_prop(training_instances)
else:
    att_props = None
model = MLP(tagset_sizes=tag_set_sizes,
                   char_hidden_dim=options.char_hidden_dim,
                   hidden_dim=options.hidden_dim,
                   word_embeddings=word_embeddings,
                   we_update = options.we_update,
                   use_char_rnn=options.use_char_rnn,
                   charset_size=len(c2i),
                   lowercase_words=options.lowercase_words,
                   vocab_size=len(w2i),
                   att_props=att_props,
                   word_embedding_dim=DEFAULT_WORD_EMBEDDING_SIZE)

trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9, 0.1)
logging.info("Training Algorithm: {}".format(type(trainer)))

logging.info("Number training instances: {}".format(len(training_instances)))
logging.info("Number dev instances: {}".format(len(dev_instances)))

dev_accs = []
dev_mic_f1s = []
for epoch in xrange(int(options.num_epochs)):
    bar = progressbar.ProgressBar()
    random.shuffle(training_instances)
    train_loss = 0.0
    train_correct = Counter()
    train_total = Counter()

    if options.debug:
        train_instances = training_instances[0:int(len(training_instances)/10)]
    else:
        train_instances = training_instances
    
    for idx,instance in enumerate(bar(train_instances)):
        gold_tags = instance.tags
        for att in model.attributes:
            if att not in instance.tags or len(instance.tags[att]) == 0:
                gold_tags[att] = [t2is[att][NONE_TAG]]
        loss_exprs = model.loss(instance.word, gold_tags)
        loss_expr = dy.esum(loss_exprs.values())
        
        loss = loss_expr.scalar_value()
        
        # Bail if loss is NaN
        if math.isnan(loss):
            assert False, "NaN occured"

        train_loss += loss

        # Do backward pass and update parameters
        loss_expr.backward()
        trainer.update()

    logging.info("\n")
    logging.info("Epoch {} complete".format(epoch + 1))
    trainer.update_epoch(1)
    print trainer.status()

    # Evaluate dev data
    dev_loss = 0.0
    dev_correct = Counter()
    dev_total = Counter()
    bar = progressbar.ProgressBar()
    f1_eval = Evaluator(m = 'att')
    if options.debug:
        d_instances = dev_instances[0:int(len(dev_instances)/5)]
    else:
        d_instances = dev_instances
    pred_strings = []
    for instance in bar(d_instances):
        gold_tags = instance.tags
        for att in model.attributes:
            if att not in instance.tags or len(instance.tags[att]) == 0:
                gold_tags[att] = [t2is[att][NONE_TAG]]
        losses = model.loss(instance.word, gold_tags)
        total_loss = sum([l.scalar_value() for l in losses.values()]) # TODO or average
        out_tags_set = model.tag_word(instance.word)
        
        gold_string = utils.single_morphotag_string(i2ts, gold_tags)
        obs_string = utils.single_morphotag_string(i2ts, out_tags_set)
        f1_eval.add_instance(utils.split_tagstring(gold_string, has_pos=False),\
                                utils.split_tagstring(obs_string, has_pos=False))
        
        # compute accuracy per attribute
        for att, gtags in gold_tags.items():
            if att == POS_KEY: continue
            out_tags = out_tags_set[att]
            if gtags == out_tags:
                if gtags == [t2is[att][NONE_TAG]]: continue
                dev_correct[att] += 1
            dev_total[att] += 1
        
        dev_loss += total_loss
        pred_strings.append("{}\t{}\t{}".format(instance.word, gold_string, obs_string).encode('utf8'))
        
    if (epoch + 1) % 50 == 0 or epoch + 1 == options.num_epochs:
        with open("{}/devout_epoch-{:03d}.txt".format(options.log_dir, epoch + 1), 'w') as dev_writer:
            dev_writer.write("\n".join(pred_strings) + "\n")

    d_acc = sum(dev_correct.values()) / sum(dev_total.values())
    dev_accs.append(FLOAT_FORMAT.format(d_acc))
    dev_mic_f1s.append(FLOAT_FORMAT.format(f1_eval.mic_f1()))
    for attr in t2is.keys():
        if attr == POS_KEY: continue
        logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
    logging.info("Total attribute F1s: {} micro, {} macro".format(f1_eval.mic_f1(), f1_eval.mac_f1()))
    
    train_loss = train_loss / len(train_instances)
    dev_loss = dev_loss / len(d_instances)
    logging.info("Train Loss: {}".format(train_loss))
    logging.info("Dev Loss: {}".format(dev_loss))
    
    # Serialize model
    if (epoch + 1) % 50 == 0 or epoch + 1 == options.num_epochs:
        new_model_file_name = "{}/model_epoch-{:03d}.bin".format(options.log_dir, epoch + 1)
        logging.info("Saving model to {}".format(new_model_file_name))
        model.save(new_model_file_name) # TODO also save non-internal model stuff like mappings

logging.info("Dev accuracies:\t{}".format("\t".join(dev_accs)))
logging.info("Dev Micro F1s:\t{}".format("\t".join(dev_mic_f1s)))
print "Final accuracy: {}".format(dev_accs[-1])
print "Final Micro F1: {}".format(dev_mic_f1s[-1])

exit()

####################################################
#### TODO re-write test part, make easy outputs ####
####################################################

# Evaluate test data (once)
logging.info("\n")
logging.info("Number test instances: {}".format(len(test_instances)))
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
            
        gold_strings = utils.single_morphotag_string(i2ts, gold_tags)
        obs_strings = utils.single_morphotag_string(i2ts, out_tags_set)
        for g, o in zip(gold_strings, obs_strings):
            f1_eval.add_instance(utils.split_tagstring(g, has_pos=True), utils.split_tagstring(o, has_pos=True))
        for att, tags in gold_tags.items():
            out_tags = out_tags_set[att]

            oov_strings = []
            for word, gold, out in zip(instance.sentence, tags, out_tags):
                if gold == out:
                    test_correct[att] += 1
                else:
                    # Got the wrong tag
                    total_wrong[att] += 1
                    if i2w[word] not in training_vocab:
                        total_wrong_oov[att] += 1
                        oov_strings.append("OOV")
                    else:
                        oov_strings.append("")
                
                if i2w[word] not in training_vocab:
                    test_oov_total[att] += 1
            test_total[att] += len(tags)
        test_writer.write(("\n"
                         + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                     gold_strings, obs_strings, oov_strings)])
                         + "\n").encode('utf8'))
        

logging.info("POS Test Accuracy: {}".format(test_correct[POS_KEY] / test_total[POS_KEY]))
for attr in t2is.keys():
    if attr != POS_KEY:
        logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), not options.pos_separate_col))

logging.info("Total test tokens: {}, Total test OOV: {}, % OOV: {}".format(test_total[POS_KEY], test_oov_total[POS_KEY], test_oov_total[POS_KEY] / test_total[POS_KEY]))
