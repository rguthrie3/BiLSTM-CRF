from __future__ import division
from collections import Counter, defaultdict
from codecs import open
from evaluate_morphotags import Evaluator
from utils import split_tagstring

import collections
import argparse
import random
import cPickle
import os
import math
import numpy as np

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

# init F1 calculators
curr_eval = Evaluator()
dict_removed_eval = Evaluator()
oracle_eval = Evaluator()
posonly_eval = Evaluator()

# load dev files
# calculate micro F1 for:
# current (=validation)
# with pos_atts data from dev POS (remove unknowns)
# with pos_atts data from pos-only dev POS (usually better -> pipeline)
# with pos_atts data from gold POS (oracle)
valid = 0
joint_good = 0
posonly_good = 0
dev_pairs_not_in_train = 0
with open(options.results, "r", encoding="utf-8") as dev_results:
    with open(options.pos_results, "r", encoding="utf-8") as pos_dev_results:
        for res, pos_res in zip(dev_results.readlines(), pos_dev_results.readlines()):
            res_cols = res.split("\t")
            if len(res_cols) < 4: continue
            pos_res_cols = pos_res.split("\t")
            assert res_cols[0:1] == pos_res_cols[0:1]
            gold = split_tagstring(res_cols[2])
            curr_obs = split_tagstring(res_cols[4])
            dict_removed_obs = {}
            posonly_obs = {}
            oracle_obs = {}
            dict_allow_atts = pos_atts[res_cols[3]]
            posonly_atts = pos_atts[pos_res_cols[3]]
            oracle_atts = pos_atts[res_cols[1]]
            if res_cols[1] == res_cols[3]:
                joint_good += 1
            if res_cols[1] == pos_res_cols[3]:
                posonly_good += 1
            # populate obs-s (TODO: list-compify)
            for att,val in curr_obs.iteritems():
                if att in dict_allow_atts:
                    dict_removed_obs[att] = val
                if att in posonly_atts:
                    posonly_obs[att] = val
                if att in oracle_atts:
                    oracle_obs[att] = val
            for a,v in gold.iteritems():
                if a not in oracle_atts:
                    dev_pairs_not_in_train += 1
            
            # add to evaluators
            curr_eval.add_instance(gold, curr_obs)
            dict_removed_eval.add_instance(gold, dict_removed_obs)
            oracle_eval.add_instance(gold, oracle_obs)
            posonly_eval.add_instance(gold, posonly_obs)
            valid += 1

print valid
print curr_eval.mic_f1()
print dict_removed_eval.mic_f1()
print oracle_eval.mic_f1()
print posonly_eval.mic_f1()
print dev_pairs_not_in_train
print joint_good / valid
print posonly_good / valid