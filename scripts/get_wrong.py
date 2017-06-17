from __future__ import division
from codecs import open
from utils import split_tagstring
from evaluate_morphotags import Evaluator
import scipy.stats
import random
import numpy as np

def word(cols):
    return cols[0].strip()

def gold_pos(cols):
    return cols[1].strip()

def pred_pos(cols):
    return cols[3].strip()

def gold_atts(cols):
    return split_tagstring(cols[2].strip())

def pred_atts(cols):
    return split_tagstring(cols[4].strip())

def is_oov(cols):
    return len(cols[-1].strip()) > 0

### Attribute F1 - Bootstrap ###

def bootstrap_samples(set_size, n):
    return [[random.randint(0, set_size - 1) for i in xrange(set_size)] for _ in xrange(n)]

def extract_att_stats(file1, file2, n):

    # get tuples per each token annotation: gold attributes, predicted by 1st algo, predicted by 2nd algo.
    # each entry within a tuple is a dictionary from attributes to values.
    all_pairs = [(gold_atts(l1.split('\t')), pred_atts(l1.split('\t')), pred_atts(l2.split('\t')))\
                    for l1, l2 in zip(file1, file2) if len(l1.strip()) > 0]

    # normal f1 for reporting
    basef1_eval1 = Evaluator(m = 'att')
    basef1_eval2 = Evaluator(m = 'att')
    for ga, oa1, oa2 in all_pairs:
        basef1_eval1.add_instance(ga, oa1)
        basef1_eval2.add_instance(ga, oa2)
    f11 = basef1_eval1.mic_f1() * 100
    f12 = basef1_eval2.mic_f1() * 100

    # retrieve samples based on this randomization
    boot_assignments = bootstrap_samples(len(all_pairs), n)

    f1_diffs = []
    for bas in boot_assignments:

        # an Evaluator aggregates (gold, observed) tuples and finally computes F1
        f1_eval1 = Evaluator(m = 'att')
        f1_eval2 = Evaluator(m = 'att')
        for idx in bas:
            g, o1, o2 = all_pairs[idx]
            f1_eval1.add_instance(g, o1)
            f1_eval2.add_instance(g, o2)
        f1_diffs.append(f1_eval1.mic_f1() - f1_eval2.mic_f1())
    diff_arr = np.array(f1_diffs)

    # compute p-value
    return f11, f12, scipy.stats.norm.cdf(diff_arr.mean() / diff_arr.std())

### POS tagging - Wilcoxon ###

def wil_p(wrong1, wrong2):
    return scipy.stats.wilcoxon(([1] * wrong1) + ([-1] * wrong2))[1]

def extract_pos_stats(file1, file2):
    wrong_poss = []
    for l1, l2 in zip(file1, file2):
        if len(l1.strip()) == 0:
            assert len(l1.strip()) == 0
            continue

        cols1 = l1.split('\t')
        cols2 = l2.split('\t')

        assert word(cols1) == word(cols2)
        assert gold_pos(cols1) == gold_pos(cols2)
        assert is_oov(cols1) == is_oov(cols2)
        pgold = gold_pos(cols1)
        isoov = is_oov(cols1)

        ppred1 = pred_pos(cols1)
        ppred2 = pred_pos(cols2)

        if ppred2 != pgold and ppred1 == pgold:
            wrong_poss.append("\t".join([word(cols1), pgold, ppred1, ppred2, str(isoov)]))

    return wrong_poss

debug = False

base_format = "rerun_full_logs/log-{}-rerun-noseq-pginit-{}char-05dr/testout.txt"
outtype = 'pos'

oov = True
#oov = False

boots_n = 100
lg = 'hi'
with open("wrongs-hi-{}-no-m.txt".format(outtype),"w","utf-8") as outfile_wo:
    with open("wrongs-hi-{}-tag-both.txt".format(outtype),"w","utf-8") as outfile_w:
        nofile = open(base_format.format(lg, "no"), "r", "utf-8").readlines()
        mfile = open(base_format.format(lg, "m"), "r", "utf-8").readlines()
        tagfile = open(base_format.format(lg, "tag"), "r", "utf-8").readlines()
        bothfile = open(base_format.format(lg, "both"), "r", "utf-8").readlines()
        if outtype == 'pos':
            wrong_poss_wo = extract_pos_stats(nofile, mfile)
            wrong_poss_w = extract_pos_stats(tagfile, bothfile)
            outfile_wo.write("\n".join(wrong_poss_wo))
            outfile_w.write("\n".join(wrong_poss_w))
