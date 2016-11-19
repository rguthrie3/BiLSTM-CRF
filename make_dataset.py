"""
Reads in raw text datasets turning them into lists of indices for use in training the model.
Expects text of the form
word###tag
word###tag
...
=== (separate sentences with this token)
word/tag
word/tag
...

Output a cPickle file of a dict with the following elements
training_instances: List of (sentence, tags) for training data
dev_instances
test_instances
w2i: Dict mapping words to indices
t2i: Dict mapping tags to indices
"""

 
import codecs
import argparse
import cPickle
import collections

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

def read_file(filename, w2i, t2i):
    """
    Read in a dataset and turn it into a list of instances.
    Modifies the w2i and t2i dicts, adding new words as it sees them.
    """
    instances = []
    with codecs.open(filename, "r", "utf-8") as f:
        sentence = []
        tags     = []
        for line in f:
            line = line.rstrip().lstrip()
            if line == "===":
                instances.append(Instance(sentence, tags))
                sentence = []
                tags     = []
            else:
                split     = line.split("###")
                word, tag = split
                if word not in w2i:
                    w2i[word] = len(w2i)
                if tag not in t2i:
                    t2i[tag] = len(t2i)
                sentence.append(w2i[word])
                tags.append(t2i[tag])
    return instances


parser = argparse.ArgumentParser()
parser.add_argument("--training-data", required=True, dest="training_data", help="Training data .txt file")
parser.add_argument("--dev-data", required=True, dest="dev_data", help="Development data .txt file")
parser.add_argument("--test-data", required=True, dest="test_data", help="Test data .txt file")
parser.add_argument("-o", required=True, dest="output", help="Output filename (.pkl)")
parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.txt", help="Text file containing all of the words in \
                    the train/dev/test data to use in outputting embeddings")
options = parser.parse_args()

w2i    = {}
t2i    = {}
output = {}
output["training_instances"] = read_file(options.training_data, w2i, t2i)
output["dev_instances"]      = read_file(options.dev_data, w2i, t2i)
output["test_instances"]     = read_file(options.test_data, w2i, t2i)
output["w2i"]                = w2i
output["t2i"]                = t2i

with open(options.output, "w") as outfile:
    cPickle.dump(output, outfile)

with codecs.open(options.vocab_file, "w", "utf-8") as vocabfile:
    for word in w2i.keys():
        vocabfile.write(word + "\n")
