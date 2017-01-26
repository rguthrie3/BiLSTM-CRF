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

if options.debug: print pos_atts

# init F1 calculators
curr_eval = Evaluator(m = 'att')
dict_removed_eval = Evaluator(m = 'att')
oracle_eval = Evaluator(m = 'att')
posonly_eval = Evaluator(m = 'att')

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
            
            gold_pos = res_cols[1]
            obs_pos = res_cols[3]
            posonly_pos = pos_res_cols[3]
            
            gold = split_tagstring(res_cols[2])
            curr_obs = split_tagstring(res_cols[4])
            
            dict_allow_atts = pos_atts[obs_pos]
            posonly_atts = pos_atts[posonly_pos]
            oracle_atts = pos_atts[gold_pos]
            if gold_pos == obs_pos:
                joint_good += 1 # validate log result
            if gold_pos == posonly_pos:
                posonly_good += 1 # validate log result
            
            dict_removed_obs = {att:val for att, val in curr_obs.iteritems() if att in dict_allow_atts}
            posonly_obs = {att:val for att, val in curr_obs.iteritems() if att in posonly_atts}
            oracle_obs = {att:val for att, val in curr_obs.iteritems() if att in oracle_atts}
            
            unseen = [a for a in gold if a not in oracle_atts]
            if len(unseen) > 0 and options.debug: print "Unseen", gold_pos, unseen
            dev_pairs_not_in_train += len(unseen)
            
            # add to evaluators
            curr_eval.add_instance(gold, curr_obs)
            dict_removed_eval.add_instance(gold, dict_removed_obs)
            oracle_eval.add_instance(gold, oracle_obs)
            posonly_eval.add_instance(gold, posonly_obs)
            valid += 1

if options.debug: print "Tokens:", valid

print "\n", "Joint POS acc (validation):", joint_good / valid
print "POS-only acc (validation):", posonly_good / valid, "\n"
print "Joint F1 (validation):", curr_eval.mic_f1()
print "Post-processed F1:", dict_removed_eval.mic_f1()
print "POS-only processed F1:", posonly_eval.mic_f1()
print "Oracle F1:", oracle_eval.mic_f1()
print "Dev POS-att pairs not in training set:", dev_pairs_not_in_train