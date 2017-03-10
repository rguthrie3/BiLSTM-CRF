from _collections import defaultdict

"""
Reads in CONLL files to make the dataset
Output a cPickle file of a dict with the following elements
training_instances: List of (word, dict of morphotags) for training data
dev_instances
test_instances
w2i: Dict mapping words to indices
t2is: Dict mapping attribute types (POS / morpho) to dicts from tags to indices
c2i: Dict mapping characters to indices
"""

import codecs
import argparse
import cPickle
import collections
import morfessor
from utils import split_tagstring

Instance = collections.namedtuple("Instance", ["word", "tags"])

DONT_KEEP = "<DONT_KEEP_WORD>"

NONE_TAG = "<NONE>"
PADDING_CHAR = "<*>"
POS_KEY = "POS"
MORPH_KEY = "MORPH"

def read_file(filename, t2is, c2i, forbidden_words=[]):
    """
    Read in a dataset and turn it into a list of instances.
    Modifies the t2is and c2i dicts, adding new attributes/tags/chars 
    as it sees them.
    """        
    mtsets = defaultdict(lambda: defaultdict(set))
    with codecs.open(filename, "r", "utf-8") as f:
        tags = defaultdict(list)
        for i, line in enumerate(f):
            if line.startswith("#") or line.isspace(): # comment / new sentence
                continue
            else:
                data = line.split("\t")
                if '-' in data[0]: # Italian has contractions on a separate line, we don't want to include them also
                    continue
                word = data[1]
                if word in forbidden_words: continue
                postag = data[3] if options.ud_tags else data[4]
                morphotags = split_tagstring(data[5], uni_key=options.flat_morphotags)
                word_seen = word in mtsets
                keep_word = True
                
                # populate unseen tags and chars if exist
                pt2i = t2is[POS_KEY]
                if postag not in pt2i:
                    pt2i[postag] = len(pt2i)
                for c in word:
                    if c not in c2i:
                        c2i[c] = len(c2i)
                
                # populate unseen morphotags if exist
                if options.flat_morphotags:
                    for mtag in morphotags:
                        mt2i = t2is[MORPH_KEY]
                        if mtag not in mt2i:
                            mt2i[mtag] = len(mt2i)
                else:
                    for att, vals in morphotags.items():
                        mt2i = t2is[att]
                        for v in vals:
                            if v not in mt2i:
                                mt2i[v] = len(mt2i)
                
                mts = mtsets[word]
                for att, vals in morphotags.items():
                    curr_vals = mts[att]
                    if len(curr_vals) == 0:
                        curr_vals.update(vals)
                    elif curr_vals == vals:
                        continue
                    # we're still here? there's a conflict.
                    elif len(forbidden_words) == 0: # training set - ignore word
                        mts[DONT_KEEP] = set([True])
                    else: # dev or test set - keep intersection
                        curr_vals.intersection_update(vals)
                
    instances = []
    multitag_wordatts = 0
    # if needed, w2i can be created here.
    for w,d in mtsets.iteritems():
        if DONT_KEEP in d:
            continue
        tags = {}
        for att,val_set in d.iteritems():
            tags[att] = [t2is[att][v] for v in val_set]
            if len(val_set) > 1:
                multitag_wordatts += 1
        instances.append(Instance(w, tags))
    print "{} word-attributes with multiple tags".format(multitag_wordatts)
    return instances, mtsets.keys()
    
parser = argparse.ArgumentParser()
parser.add_argument("--training-data", required=True, dest="training_data", help="Training data .txt file")
parser.add_argument("--dev-data", required=True, dest="dev_data", help="Development data .txt file")
parser.add_argument("--test-data", required=True, dest="test_data", help="Test data .txt file")
parser.add_argument("--ud-tags", dest="ud_tags", action="store_true", help="Extract UD tags instead of original tags")
parser.add_argument("--flat-morphotags", dest="flat_morphotags", action="store_true", help="Morphosyntactic tags are flattened to single features")
parser.add_argument("-o", required=True, dest="output", help="Output filename (.pkl)")
parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.txt", help="Text file containing all of the words in \
                    the train/dev/test data to use in outputting embeddings")
options = parser.parse_args()


t2is = defaultdict(lambda: {NONE_TAG:0}) # mapping from attribute name to mapping from tag to index
c2i = {}
output = {}
output["training_instances"], train_words = read_file(options.training_data, t2is, c2i)
output["dev_instances"], dev_words = read_file(options.dev_data, t2is, c2i, train_words)
output["test_instances"], test_words = read_file(options.test_data, t2is, c2i, train_words)

# Add special tokens / tags / chars to dicts
c2i[PADDING_CHAR] = len(c2i)

output["t2is"] = {t:i_s for t, i_s in t2is.items()}
output["c2i"] = c2i

with open(options.output, "w") as outfile:
    cPickle.dump(output, outfile)

with codecs.open(options.vocab_file, "w", "utf-8") as vocabfile:
    for word in train_words + dev_words + test_words:
        vocabfile.write(word + "\n")
