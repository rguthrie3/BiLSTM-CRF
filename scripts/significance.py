from __future__ import division
from codecs import open
from utils import split_tagstring
from evaluate_morphotags import Evaluator
import scipy.stats
from scipy.stats import chi2
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

def header(outtype):
    if outtype == 'pos':
        return "Language\tNone acc\tTag acc\tMim acc\tBoth acc"
    elif outtype == 'att':
        return "Language\tNone F1\tTag F1\tMim F1\tBoth F1"
    elif outtype == 'att-raw':
        return "Language\tTag F1\tMim F1\tp"
    else: # 'pos-raw' or 'pos-sent'
        return "Language\tTag acc\tMim acc\tp"

### Attribute F1 ###

def bootstrap_samples(set_size, n):
    return [[random.randint(0, set_size - 1) for i in xrange(set_size)] for _ in xrange(n)]

def extract_bootstrap_att_stats(file1, file2, n):
    '''
    get tuples per each token annotation: gold attributes, predicted by 1st algo, predicted by 2nd algo.
    each entry within a tuple is a dictionary from attributes to values.
    algo from http://nlp.cs.berkeley.edu/pubs/BergKirkpatrick-Burkett-Klein_2012_Significance_paper.pdf
    '''
    all_pairs = [(gold_atts(l1.split('\t')), pred_atts(l1.split('\t')), pred_atts(l2.split('\t')))\
                    for l1, l2 in zip(file1, file2) if len(l1.strip()) > 0]

    # normal f1 for reporting
    basef1_eval1 = Evaluator(m = 'att')
    basef1_eval2 = Evaluator(m = 'att')
    for ga, oa1, oa2 in all_pairs:
        basef1_eval1.add_instance(ga, oa1)
        basef1_eval2.add_instance(ga, oa2)
    f11 = basef1_eval1.mic_f1()
    f12 = basef1_eval2.mic_f1()
    base_delta = np.abs(f11 - f12)

    # retrieve samples based on this randomization
    boot_assignments = bootstrap_samples(len(all_pairs), n)

    # f1_diffs = [] # wrong
    s = 0
    for bas in boot_assignments:

        # an Evaluator aggregates (gold, observed) tuples and finally computes F1
        f1_eval1 = Evaluator(m = 'att')
        f1_eval2 = Evaluator(m = 'att')
        for idx in bas:
            g, o1, o2 = all_pairs[idx]
            f1_eval1.add_instance(g, o1)
            f1_eval2.add_instance(g, o2)
        delta = f1_eval1.mic_f1() - f1_eval2.mic_f1()
        if np.abs(delta) > (base_delta * 2):
            s += 1
        # f1_diffs.append(f1_eval1.mic_f1() - f1_eval2.mic_f1()) # wrong
    # diff_arr = np.array(f1_diffs) # wrong

    # compute p-value
    return f11 * 100, f12 * 100, s / n
    # return f11, f12, scipy.stats.norm.cdf(diff_arr.mean() / diff_arr.std()) # wrong

def extract_sentence_level_att_stats(file1, file2):
    total_sens = 0
    senlen = 0
    f1s1 = []
    f1s2 = []
    f1_eval1 = Evaluator(m = 'att')
    f1_eval2 = Evaluator(m = 'att')
    for l1, l2 in zip(file1, file2):
        if len(l1.strip()) == 0:
            assert len(l2.strip()) == 0
            if senlen == 0: continue
            total_sens += 1
            # add diff to aggregator
            f1s1.append(f1_eval1.mic_f1())
            f1s2.append(f1_eval2.mic_f1())

            # restart caches
            f1_eval1 = Evaluator(m = 'att')
            f1_eval2 = Evaluator(m = 'att')
            senlen = 0
            continue

        senlen += 1

        cols1 = l1.split('\t')
        cols2 = l2.split('\t')

        assert word(cols1) == word(cols2)
        assert gold_atts(cols1) == gold_atts(cols2)
        g = gold_atts(cols1)
        o1 = pred_atts(cols1)
        o2 = pred_atts(cols2)
        f1_eval1.add_instance(g, o1)
        f1_eval2.add_instance(g, o2)

    wil = scipy.stats.wilcoxon(x = f1s1, y = f1s2, zero_method='wilcox')[1]
    return np.average(f1s1), np.average(f1s2), total_sens, wil

### POS tagging ###

def extract_sentence_level_pos_stats(file1, file2):
    total_sens = 0
    accs1 = []
    accs2 = []
    acc_diffs = []
    senlen = 0
    corr1 = 0
    corr2 = 0
    for l1, l2 in zip(file1, file2):
        if len(l1.strip()) == 0:
            assert len(l2.strip()) == 0
            if senlen == 0: continue
            total_sens += 1
            # add diff to aggregator
            acc1 = corr1 / senlen
            acc2 = corr2 / senlen
            accs1.append(acc1)
            accs2.append(acc2)
            if corr1 != corr2:
                acc_diffs.append(acc1 - acc2)

            # restart caches
            senlen = 0
            corr1 = 0
            corr2 = 0
            continue

        senlen += 1

        cols1 = l1.split('\t')
        cols2 = l2.split('\t')

        assert word(cols1) == word(cols2)
        assert gold_pos(cols1) == gold_pos(cols2)
        assert is_oov(cols1) == is_oov(cols2)
        pgold = gold_pos(cols1)

        ppred1 = pred_pos(cols1)
        ppred2 = pred_pos(cols2)

        #accuracy in percent
        if ppred1 == pgold: corr1 += 1
        if ppred2 == pgold: corr2 += 1

    #wil = scipy.stats.wilcoxon(acc_diffs)[1]
    wil = scipy.stats.wilcoxon(x = accs1, y = accs2, zero_method='wilcox')[1]
    return np.average(accs1), np.average(accs2), total_sens, wil

def mcnemar(wrong1, wrong2):
    chi_sq = ((wrong1 - wrong2) ** 2) / (wrong1 + wrong2)
    return 1.0 - chi2.cdf(chi_sq, 1)

#def wil(wrong1, wrong2):
    #return scipy.stats.wilcoxon(([1] * wrong1) + ([-1] * wrong2))[1]

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
            assert len(l2.strip()) == 0
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
            wrong1, wrong2, total, mcnemar(wrong1, wrong2),\
            wrong_oov1, wrong_oov2, mcnemar(wrong_oov1, wrong_oov2)

debug = False

# langs = ['fa', 'hi', 'en', 'es', 'it', 'da', 'he', 'sv', 'bg', 'cs', 'lv', 'hu', 'tr', 'ta', 'ru', 'vi'] # order by %OOV
langs = ['ta', 'lv', 'vi', 'hu', 'tr', 'bg', 'sv', 'ru', 'da', 'fa', 'he', 'en', 'hi', 'it', 'es', 'cs'] # order by training data

#base_format = "logs_token_exp_sign/log-{}-10k-noseq-pginit-{}char-05dr/testout.txt"
#base_format = "rerun_full_logs/log-{}-rerun-noseq-pginit-{}char-05dr/testout.txt"
base_format = "logs_token_exp_sign_5k/log-{}-5k-noseq-pginit-{}char-05dr/testout.txt"

testname = "mvtag-sents-5k"

#bar = 0.01
bar = 0.05

#outtype = 'pos-raw'
#outtype = 'pos-sent'
outtype = 'att-raw'
#outtype = 'pos'
#outtype = 'att-boot'

#oov = True
oov = False

boots_n = 100

with open("logs_token_exp-{}-sign-{}{}-{}.txt".format(testname, outtype, "-oov" if oov else "", bar),"w","utf-8") as outfile:
    outfile.write(header(outtype) + "\n")
    for lg in langs:
        if outtype == 'att' and lg == 'vi': continue

        # mchar vs. nochar
        nofile = open(base_format.format(lg, "no"), "r", "utf-8").readlines()
        mfile = open(base_format.format(lg, "m"), "r", "utf-8").readlines()
        if outtype == 'pos':
            accn, accm, accnv, accmv, wn, wm, _, wilwo, wno, wmo, wilwov = extract_pos_stats(nofile, mfile)
        elif outtype == 'att-boot':
            f1n, f1m, atwo = extract_bootstrap_att_stats(nofile, mfile, boots_n)

        if debug: print lg, "w/o", wn, wm, wilwo, wno, wmo, wilwov, atwo

        # bothchar vs. tagchar
        tagfile = open(base_format.format(lg, "tag"), "r", "utf-8").readlines()
        bothfile = open(base_format.format(lg, "both"), "r", "utf-8").readlines()
        if outtype == 'pos':
            acct, accb, acctv, accbv, wt, wb, _, wilw, wto, wbo, wilwv = extract_pos_stats(tagfile, bothfile)
        elif outtype == 'att-boot':
            f1t, f1b, atw = extract_bootstrap_att_stats(tagfile, bothfile, boots_n)

        if outtype.endswith('raw'):
            if outtype.startswith('pos'):
                acct, accm, _, _, wt, wm, tot, wilw, _, _, _ = extract_pos_stats(tagfile, mfile)
            else: # att-raw
                f1t, f1m, tot, atw = extract_sentence_level_att_stats(tagfile, mfile)

        if outtype == 'pos-sent':
            acct, accm, tot, wilw = extract_sentence_level_pos_stats(tagfile, mfile)

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
        elif outtype == 'att-boot':
            m_sign = "*" if atwo < bar else ""
            b_sign = "*" if atw < bar else ""
            outfile.write("{}\t{:.2f}\t{:.2f}\t{:.2f}{}\t{:.2f}{}\n".format(lg,f1n,f1t,f1m,m_sign,f1b,b_sign))
            print "finished {}".format(lg)
        elif outtype == 'pos-raw':
            outfile.write("{}\t{:.4f}\t{:.4f}\t{:.12f}\t{}\t{}\t{}\n".format(lg,acct,accm,wilw,wt,wm,tot))
        elif outtype == 'pos-sent':
            outfile.write("{}\t{:.4f}\t{:.4f}\t{:.12f}\t{}\n".format(lg,acct,accm,wilw,tot))
        else: # att-raw
            outfile.write("{}\t{:.4f}\t{:.4f}\t{:.12f}\t{}\n".format(lg,f1t,f1m,atw,tot))


        #outfile.write("\t".join([lg, str(wilwo < bar), str(wilw < bar),\
            #str(wilwov < bar), str(wilwv < bar), str(atwo < bar), str(atw < bar)]) + "\n")
