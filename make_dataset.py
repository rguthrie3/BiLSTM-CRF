# TODO right now there is kinda a chicken and egg problem.
# this file outputs the vocab list which would be used to get the morpheme segmentations list
# could just use Morfessor but that would require dependence on Morfessor and might be obnoxious to change
# later
# Probably just add a flag like --output-vocab-file that will just dump the words and exit
from _collections import defaultdict

"""
Reads in CONLL files to make the dataset
Output a cPickle file of a dict with the following elements
training_instances: List of (sentence, tags) for training data
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

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

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


def read_file(filename, w2i, t2is, c2i):
    """
    Read in a dataset and turn it into a list of instances.
    Modifies the w2i, t2is and c2i dicts, adding new words/attributes/tags/chars 
    as it sees them.
    """
    
    # populate mandatory t2i tables
    if "POS" not in t2is:
        t2is["POS"] = {}
    if options.flat_morphotags and "MORPH" not in t2is:
        t2is["MORPH"] = {"NONE":0}
    
    instances = []
    vocab_counter = collections.Counter()
    with codecs.open(filename, "r", "utf-8") as f:
        sentence = []
        tags = defaultdict(list)
        for i, line in enumerate(f):
            if line.startswith("#"):
                continue # Some files like Italian have comments
            elif line.isspace():
                # Reached the end of a sentence
                slen = len(sentence)
                for seq in tags.values():
                    if len(seq) < slen:
                        seq.extend([0] * (slen - len(seq)))
                instances.append(Instance(sentence, tags))
                sentence = []
                tags = defaultdict(list)
            else:
                data = line.split("\t")
                if '-' in data[0]: # Italian has contractions on a separate line, we don't want to include them also
                    continue
                word = data[1]
                postag = data[3] if options.ud_tags else data[4]
                morphotags = split_tagstring(data[5], uni_key=options.flat_morphotags) if options.morphotags else {}
                vocab_counter[word] += 1
                if word not in w2i:
                    w2i[word] = len(w2i)
                pt2i = t2is["POS"]
                if postag not in pt2i:
                    pt2i[postag] = len(pt2i)
                for c in word:
                    if c not in c2i:
                        c2i[c] = len(c2i)
                if options.flat_morphotags:
                    for mtag in morphotags:
                        mt2i = t2is["MORPH"]
                        if mtag not in mt2i:
                            mt2i[mtag] = len(mt2i)
                else:
                    for key, val in morphotags.items():
                        if key not in t2is:
                            t2is[key] = {"NONE":0}
                        mt2i = t2is[key]
                        if val not in mt2i:
                            mt2i[val] = len(mt2i)
                sentence.append(w2i[word])
                tags["POS"].append(t2i[postag])
                if options.flat_morphotags:
                    tags["MORPH"].append([mt2i[t] for t in morphotags.items()])
                else:
                    for k,v in morphotags.items():
                        mtags = tags[k]
                        if len(mtags) == 0 and len(sentence) > 1:
                            # tag not seen in sentence, need to pad backwards
                            mtags.extend([0] * (len(sentence) - 1))                            
                        mtags.append(t2is[k][v])
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
parser.add_argument("--flat-morphotags", dest="flat_morphotags", default=False, help="Morphosyntactic tags are flattened to single features")
parser.add_argument("--morpheme-segmentations", dest="morpheme_segmentations", help="Morpheme segmentations file")
parser.add_argument("-o", required=True, dest="output", help="Output filename (.pkl)")
parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.txt", help="Text file containing all of the words in \
                    the train/dev/test data to use in outputting embeddings")
options = parser.parse_args()


w2i = {} # mapping from word to index
t2is = {} # mapping from attribute name to mapping from tag to index
c2i = {}
output = {}
output["training_instances"], output["training_vocab"] = read_file(options.training_data, w2i, t2is, c2i)
output["dev_instances"], output["dev_vocab"] = read_file(options.dev_data, w2i, t2is, c2i)
output["test_instances"], output["test_vocab"] = read_file(options.test_data, w2i, t2is, c2i)
if options.morpheme_segmentations is not None:
    m2i = {}
    output["morpheme_segmentations"] = read_morpheme_segmentations(options.morpheme_segmentations, w2i, m2i)
    output["m2i"] = m2i

# Add special tokens / tags / chars to dicts
w2i["<UNK>"] = len(w2i)
for t2i in t2is:
    t2i["<START>"] = len(t2i)
    t2i["<STOP>"] = len(t2i)
c2i["<*>"] = len(c2i) # padding char

output["w2i"] = w2i
output["t2is"] = t2is
output["c2i"] = c2i

with open(options.output, "w") as outfile:
    cPickle.dump(output, outfile)

with codecs.open(options.vocab_file, "w", "utf-8") as vocabfile:
    for word in w2i.keys():
        vocabfile.write(word + "\n")
