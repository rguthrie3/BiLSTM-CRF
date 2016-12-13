# TODO right now there is kinda a chicken and egg problem.
# this file outputs the vocab list which would be used to get the morpheme segmentations list
# could just use Morfessor but that would require dependence on Morfessor and might be obnoxious to change
# later
# Probably just add a flag like --output-vocab-file that will just dump the words and exit

"""
Reads in CONLL files to make the dataset
Output a cPickle file of a dict with the following elements
training_instances: List of (sentence, tags) for training data
dev_instances
test_instances
w2i: Dict mapping words to indices
t2i: Dict mapping tags to indices
c2i: Dict mapping characters to indices
"""

 
import codecs
import argparse
import cPickle
import collections
import morfessor
from utils import split_tagstring

Instance = collections.namedtuple("Instance", ["sentence", "tags", "mtags"])

def read_morpheme_segmentations(filename, w2i, m2i):
    segmentations = {}
    with codecs.open(filename, "r", "utf-8") as f:
        for line in f:
            split = line.rstrip().lstrip().split()
            word = split[0]
            if word in w2i:
                morphemes = split[1:]
                for m in morphemes:
                    if m not in m2i:
                        m2i[m] = len(m2i)
                segmentations[w2i[word]] = [ m2i[m] for m in morphemes ]
    return segmentations


def read_file(filename, w2i, t2i, mt2i, c2i):
    """
    Read in a dataset and turn it into a list of instances.
    Modifies the w2i, t2i and c2i dicts, adding new words/tags/chars 
    as it sees them.
    """
    instances = []
    vocab_counter = collections.Counter()
    with codecs.open(filename, "r", "utf-8") as f:
        sentence = []
        tags = []
        mtags = []
        for i, line in enumerate(f):
            if line.startswith("#"):
                continue # Some files like Italian have comments
            elif line.isspace():
                # Reached the end of a sentence
                instances.append(Instance(sentence, tags, mtags))
                sentence = []
                tags = []
                mtags = []
            else:
                data = line.split("\t")
                if '-' in data[0]: # Italian has contractions on a separate line, we don't want to include them also
                    continue
                word = data[1]
                tag = data[3] if options.ud_tags else data[4]
                morphotags = split_tagstring(data[5], uni_key=True) if options.morphotags else {}
                vocab_counter[word] += 1
                if word not in w2i:
                    w2i[word] = len(w2i)
                if tag not in t2i:
                    t2i[tag] = len(t2i)
                for c in word:
                    if c not in c2i:
                        c2i[c] = len(c2i)
                for mtag in morphotags:
                    if mtag not in mt2i:
                        mt2i[mtag] = len(mt2i)
                sentence.append(w2i[word])
                tags.append(t2i[tag])
                mtags.append([mt2i[t] for t in morphotags])
    return instances, vocab_counter

# def read_file(filename, w2i, t2i):
#     instances = []
#     vocab_counter = collections.Counter()
#     with codecs.open(filename, "r", "utf-8") as f:
#         sentence = []
#         tags = []
#         for i, line in enumerate(f):
#             if line == "=====\n":
#                 # Reached the end of a sentence
#                 instances.append(Instance(sentence, tags))
#                 sentence = []
#                 tags = []
#             else:
#                 data = line.split("/")
#                 if len(data) > 2:
#                     word = '/'.join(data[:-2])
#                 else:
#                     word = data[0]
#                 tag = data[-1]
#                 vocab_counter[word] += 1
#                 if word not in w2i:
#                     w2i[word] = len(w2i)
#                 if tag not in t2i:
#                     t2i[tag] = len(t2i)
#                 sentence.append(w2i[word])
#                 tags.append(t2i[tag])
#     return instances, vocab_counter

parser = argparse.ArgumentParser()
parser.add_argument("--training-data", required=True, dest="training_data", help="Training data .txt file")
parser.add_argument("--dev-data", required=True, dest="dev_data", help="Development data .txt file")
parser.add_argument("--test-data", required=True, dest="test_data", help="Test data .txt file")
parser.add_argument("--ud-tags", dest="ud_tags", action="store_true", help="Extract UD tags instead of original tags")
parser.add_argument("--morphotags", dest="morphotags", default=False, help="Add morphosyntactic tags to dataset")
parser.add_argument("--morpheme-segmentations", dest="morpheme_segmentations", help="Morpheme segmentations file")
parser.add_argument("-o", required=True, dest="output", help="Output filename (.pkl)")
parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.txt", help="Text file containing all of the words in \
                    the train/dev/test data to use in outputting embeddings")
options = parser.parse_args()


w2i = {} # mapping from word to index
t2i = {} # mapping from POS tag to index
mt2i = {} # mapping from Morphosyntactic tag name + value to index (TODO: also allow nested mapping)
c2i = {}
output = {}
output["training_instances"], output["training_vocab"] = read_file(options.training_data, w2i, t2i, mt2i, c2i)
output["dev_instances"], output["dev_vocab"] = read_file(options.dev_data, w2i, t2i, mt2i, c2i)
output["test_instances"], output["test_vocab"] = read_file(options.test_data, w2i, t2i, mt2i, c2i)
if options.morpheme_segmentations is not None:
    m2i = {}
    output["morpheme_segmentations"] = read_morpheme_segmentations(options.morpheme_segmentations, w2i, m2i)
    output["m2i"] = m2i

# Add special tokens / tags / chars to dicts
w2i["<UNK>"] = len(w2i)
t2i["<START>"] = len(t2i)
t2i["<STOP>"] = len(t2i)
c2i["<*>"] = len(c2i) # padding char

output["w2i"] = w2i
output["t2i"] = t2i
output["c2i"] = c2i
output["mt2i"] = mt2i

with open(options.output, "w") as outfile:
    cPickle.dump(output, outfile)

with codecs.open(options.vocab_file, "w", "utf-8") as vocabfile:
    for word in w2i.keys():
        vocabfile.write(word + "\n")
