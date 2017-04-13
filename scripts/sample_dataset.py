import argparse
import cPickle
import random
from collections import Counter, namedtuple
from _collections import defaultdict

Instance = namedtuple("Instance", ["sentence", "tags"])

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--output", required=True, dest="output", help="Output filename (.pkl)")
parser.add_argument("--token-size", default=10000, dest="token_size", type=int, help="Token count of training set")
options = parser.parse_args()

# read data
dataset = cPickle.load(open(options.dataset, "r"))
training_corpus_size = sum(dataset["training_vocab"].values())
i2w = { i: w for w, i in dataset["w2i"].items() }

# init new dataset
new_dataset = dataset

# sample training dataset
training_instances = dataset["training_instances"]
if training_corpus_size > options.token_size:
    random.shuffle(training_instances)
    cumulative_tokens = 0
    cutoff_index = -1
    for i,inst in enumerate(training_instances):
        cumulative_tokens += len(inst.sentence)
        if cumulative_tokens >= options.token_size:
            training_instances = training_instances[:i+1]
            break

# recompute vocab
training_vocab = Counter()
for inst in training_instances:
    for w in inst.sentence:
        training_vocab[i2w[w]] += 1

# rewrite training set atts
new_dataset["training_instances"] = training_instances
new_dataset["training_vocab"] = training_vocab

# output
with open(options.output, "w") as outfile:
    cPickle.dump(new_dataset, outfile)

