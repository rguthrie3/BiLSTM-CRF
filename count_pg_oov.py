from __future__ import division

import collections
import argparse
import cPickle
import os
import math
from codecs import open

Instance = collections.namedtuple("Instance", ["chars", "word_emb"])

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--results", required=True, dest="results", help="results file to use")
parser.add_argument("--has-atts", dest="has_atts", action="store_true", help="toggle if file has attributes")
options = parser.parse_args()

dataset = cPickle.load(open(options.dataset, "r"))
i2c = { i: c for c, i in dataset["c2i"].items() } # inverse map
test_instances = dataset["test_instances"]
test_words = [''.join([i2c[i] for i in instance.chars]) for instance in test_instances]
print "Evaluating on {} word types.".format(len(test_words))

total_tokens = 0
total_correct = 0
oov_tokens = 0
oov_correct = 0

with open(options.results) as res_file:
    for l in res_file.readlines():
        l = l.strip()
        if l.startswith("#") or len(l) == 0: continue
        cols = l.split("\t")
        word = cols[0]
        gold = cols[1]
        obs = cols[3]
        total_tokens += 1
        if gold == obs:
            total_correct += 1
        if word in test_words:
            oov_tokens += 1
            if gold == obs:
                oov_correct += 1

print "Total accuracy (sanity check): {} of {} = {}".format(total_correct, total_tokens, total_correct / total_tokens)
print "PG OOV accuracy: {} of {} = {}".format(oov_correct, oov_tokens, oov_correct / oov_tokens)
        