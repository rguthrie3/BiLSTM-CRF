import cPickle
import itertools
import codecs
import numpy as np
import matplotlib.pyplot as plt
import dynet as dy
from sklearn.metrics import confusion_matrix

NONE_TAG = "<NONE>"
POS_KEY = "POS"

class CSVLogger:

    def __init__(self, filename, columns):
        self.file = open(filename, "w")
        self.columns = columns
        self.file.write(','.join(columns) + "\n")

    def add_column(self, data):
        self.file.write(','.join([str(d) for d in data]) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class ConfusionMatrix:

    def __init__(self, classes):
        self.cm = None
        self.classes = classes

    def add(self, targets, preds):
        new_cm = confusion_matrix(targets, preds)
        if self.cm is None:
            self.cm = new_cm
        else:
            self.cm += new_cm

    def plot(self, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(self.cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        if normalize:
            self.cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]

        thresh = self.cm.max() / 2.
        for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
            plt.text(j, i, self.cm[i, j],
                     horizontalalignment="center",
                     color="white" if self.cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


def kl_div(x, y):
    sig_x = dy.logistic(x)
    exp_x = dy.exp(x)
    exp_y = dy.exp(y)
    exp_neg_x = dy.exp(-x)
    exp_neg_y = dy.exp(-y)
    shape = x.value().shape
    matrix_size = shape[0] * shape[1]
    ones = dy.reshape(dy.inputVector([1] * matrix_size), shape)
    total = dy.cmult(sig_x, dy.log(ones + exp_neg_y) - dy.log(ones + exp_neg_x)) + dy.cmult(ones - sig_x, dy.log(ones + exp_y) - dy.log(ones +exp_x))
    # now we average in a convoluted way
    total_cols = dy.reshape(total, (1, matrix_size))
    sum = dy.sum_cols(total_cols)
    return dy.cdiv(sum, dy.inputVector([matrix_size]))


def convert_instance(instance, i2w, i2t):
    sent = [ i2w[w] for w in instance.sentence ]
    tags = [ i2t[t] for t in instance.tags ]
    return sent, tags


def read_pretrained_embeddings(filename, w2i):
    word_to_embed = {}
    with codecs.open(filename, "r", "utf-8") as f:
        for line in f:
            split = line.split()
            if len(split) > 0:
                word = split[0]
                vec = split[1:]
                word_to_embed[word] = vec
    embedding_dim = len(word_to_embed[word_to_embed.keys()[0]])
    out = np.random.uniform(-0.8, 0.8, (len(w2i), embedding_dim))
    for word, embed in word_to_embed.items():
        embed_arr = np.array(embed)
        if np.linalg.norm(embed_arr) < 15.0 and word in w2i:
            # Theres a reason for this if condition.  Some tokens in ptb
            # cause numerical problems because they are long strings of the same punctuation, e.g
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! which end up having huge norms, since Morfessor will
            # segment it as a ton of ! and then the sum of these morpheme vectors is huge.
            out[w2i[word]] = np.array(embed)
    return out


def split_tagstring(s, uni_key=False):
    '''
    Returns attribute-value mapping from UD-type CONLL field
    @param uni_key: if toggled, returns attribute-value pairs as joined strings (with the '=')
    '''
    ret = [] if uni_key else {}
    if "=" not in s: # incorrect format
        return ret
    for attval in s.split('|'):
        attval = attval.strip()
        if not uni_key:
            a,v = attval.split('=')
            ret[a] = v
        else:
            ret.append(attval)
    return ret


def morphotag_strings(i2ts, tag_mapping, pos_separate_col=True):
    senlen = len(tag_mapping.values()[0])
    key_value_strs = []
    
    # j iterates along sentence, as we're building the string representations
    # in the opposite orientation as the mapping
    for j in xrange(senlen): 
        place_strs = []
        for att, seq in tag_mapping.items():
            val = i2ts[att][seq[j]]
            if pos_separate_col and att == POS_KEY:
                pos_str = val
            elif val != NONE_TAG:
                place_strs.append(att + "=" + val)
        morpho_str = "|".join(sorted(place_strs))
        if pos_separate_col:
            key_value_strs.append(pos_str + "\t" + morpho_str)
        else:
            key_value_strs.append(morpho_str)            
    return key_value_strs

def sortvals(dct):
    return [v for (k,v) in sorted(dct.items())]
