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
    wrong1 = 0
    wrong2 = 0
    wrong_oov1 = 0
    wrong_oov2 = 0
    corr1 = 0
    corr2 = 0
    total = 0
    total_oov = 0
    corrv1 = 0
    corrv2 = 0
    for l1, l2 in zip(file1, file2):
        if len(l1.strip()) == 0:
            assert len(l1.strip()) == 0
            continue

        total += 1
        cols1 = l1.split('\t')
        cols2 = l2.split('\t')

        assert word(cols1) == word(cols2)
        assert gold_pos(cols1) == gold_pos(cols2)
        assert is_oov(cols1) == is_oov(cols2)
        pgold = gold_pos(cols1)
        isoov = is_oov(cols1)

        ppred1 = pred_pos(cols1)
        ppred2 = pred_pos(cols2)

        #accuracy in percent
        if ppred1 == pgold: corr1 += 100
        if ppred2 == pgold: corr2 += 100

        # accuracy for OOV
        if isoov:
            total_oov += 1
            if ppred1 == pgold: corrv1 += 100
            if ppred2 == pgold: corrv2 += 100

        #significance
        if ppred1 == ppred2: continue

        if ppred1 != pgold and ppred2 == pgold:
            wrong1 += 1
            if isoov:
                wrong_oov1 += 1
        if ppred2 != pgold and ppred1 == pgold:
            wrong2 += 1
            if isoov:
                wrong_oov2 += 1

    return corr1/total, corr2/total, corrv1/total_oov, corrv2/total_oov,\
            wrong1, wrong2, total, wil_p(wrong1, wrong2),\
            wrong_oov1, wrong_oov2, wil_p(wrong_oov1, wrong_oov2)

debug = False

langs = ['fa', 'hi', 'en', 'es', 'it', 'da', 'he', 'sv', 'bg', 'cs', 'lv', 'hu', 'tr', 'ta', 'ru', 'vi'] # order by %OOV
# langs = ['ta', 'lv', 'vi', 'hu', 'tr', 'bg', 'sv', 'ru', 'da', 'fa', 'he', 'en', 'hi', 'it', 'es', 'cs'] # order by training data

#base_format = "logs_token_exp_sign/log-{}-10k-noseq-pginit-{}char-05dr/testout.txt"
base_format = "rerun_full_logs/log-{}-rerun-noseq-pginit-{}char-05dr/testout.txt"
#base_format = "logs_token_exp_sign_5k/log-{}-5k-noseq-pginit-{}char-05dr/testout.txt"

bar = 0.01
#bar = 0.05

#outtype = 'pos'
outtype = 'att'

oov = True
#oov = False

boots_n = 100

with open("logs_token_exp_full-sign-{}{}-{}.txt".format(outtype, "-oov" if oov else "", bar),"w","utf-8") as outfile:
    for lg in langs:
        if outtype == 'att' and lg == 'vi': continue

        # mchar vs. nochar
        nofile = open(base_format.format(lg, "no"), "r", "utf-8").readlines()
        mfile = open(base_format.format(lg, "m"), "r", "utf-8").readlines()
        if outtype == 'pos':
            accn, accm, accnv, accmv, wn, wm, _, wilwo, wno, wmo, wilwov = extract_pos_stats(nofile, mfile)
        else:
            f1n, f1m, atwo = extract_att_stats(nofile, mfile, boots_n)

        if debug: print lg, "w/o", wn, wm, wilwo, wno, wmo, wilwov, atwo

        # bothchar vs. tagchar
        tagfile = open(base_format.format(lg, "tag"), "r", "utf-8").readlines()
        bothfile = open(base_format.format(lg, "both"), "r", "utf-8").readlines()
        if outtype == 'pos':
            acct, accb, acctv, accbv, wt, wb, _, wilw, wto, wbo, wilwv = extract_pos_stats(tagfile, bothfile)
        else:
            f1t, f1b, atw = extract_att_stats(tagfile, bothfile, boots_n)

        if debug: print lg, "with", wt, wb, wilw, wto, wbo, wilwv, atw

        if outtype == 'pos':
            if oov:
                mv_sign = "*" if wilwov < bar else ""
                bv_sign = "*" if wilwv < bar else ""
                outfile.write("{}\t{:.2f}\t{:.2f}\t{:.2f}{}\t{:.2f}{}\n".format(lg,accnv,acctv,accmv,mv_sign,accbv,bv_sign))
            else:
                m_sign = "*" if wilwo < bar else ""
                b_sign = "*" if wilw < bar else ""
                outfile.write("{}\t{:.2f}\t{:.2f}\t{:.2f}{}\t{:.2f}{}\n".format(lg,accn,acct,accm,m_sign,accb,b_sign))
        else:
            m_sign = "*" if atwo < bar else ""
            b_sign = "*" if atw < bar else ""
            outfile.write("{}\t{:.2f}\t{:.2f}\t{:.2f}{}\t{:.2f}{}\n".format(lg,f1n,f1t,f1m,m_sign,f1b,b_sign))
            print "finished {}".format(lg)

        #outfile.write("\t".join([lg, str(wilwo < bar), str(wilw < bar),\
            #str(wilwov < bar), str(wilwv < bar), str(atwo < bar), str(atw < bar)]) + "\n")
