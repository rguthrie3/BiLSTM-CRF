from __future__ import division
from collections import Counter, defaultdict

import collections
import argparse
import random
import cPickle
import logging
import progressbar
import os
import math
import numpy as np

import utils


Instance = collections.namedtuple("Instance", ["sentence", "tags"])

NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
POS_KEY = "POS"
PADDING_CHAR = "<*>"

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--results", required=True, dest="results", help=".txt file with results")
parser.add_argument("--pos-results", dest="pos_results", help=".txt file with results of POS model alone")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
options = parser.parse_args()


# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = cPickle.load(open(options.dataset, "r"))
t2is = dataset["t2is"]
i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in t2is.items() }

train = dataset["training_instances"]

# populate POS -> attribute dictionary
pos_atts = defaultdict(set)
print "Analyzing training set"
for inst in train:
    for i in xrange(len(inst.sentence)):
        pos = i2ts["POS"][inst.tags["POS"][i]]
        for att,vals in inst.tags.iteritems():
            if att != "POS" and vals[i] != 0: # TODO change to var via t2is.inverse()
                pos_atts[pos].add(att)

print pos_atts

# load dev file

# load dev from pos_only

# calculate micro F1 for:
# current (=validation)
# with pos_atts data from dev POS (remove unknowns)
# with pos_atts data from pos-only dev POS (usually better -> pipeline)
# with pos_atts data from gold POS (oracle)