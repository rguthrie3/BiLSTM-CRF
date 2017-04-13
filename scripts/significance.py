from codecs import open
import scipy.stats

def word(cols):
    return cols[0].strip()

def gold_pos(cols):
    return cols[1].strip()

def pred_pos(cols):
    return cols[3].strip()

def gold_atts(cols):
    return cols[2].strip()

def pred_atts(cols):
    return cols[4].strip()

def is_oov(cols):
    return len(cols[-1].strip()) > 0

def wil_p(wrong1, wrong2):
    return scipy.stats.wilcoxon(([1] * wrong1) + ([-1] * wrong2))[1]

def extract_stats(file1, file2):
    wrong1 = 0
    wrong2 = 0
    wrong_oov1 = 0
    wrong_oov2 = 0
    total = 0
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
        if ppred1 == ppred2: continue

        if ppred1 != pgold and ppred2 == pgold:
            wrong1 += 1
            if isoov:
                wrong_oov1 += 1
        if ppred2 != pgold and ppred1 == pgold:
            wrong2 += 1
            if isoov:
                wrong_oov2 += 1

    return wrong1, wrong2, total, wil_p(wrong1, wrong2),\
            wrong_oov1, wrong_oov2, wil_p(wrong_oov1, wrong_oov2)

# langs = ['fa', 'hi', 'en', 'es', 'it', 'da', 'he', 'sv', 'bg', 'cs', 'lv', 'hu', 'tr', 'ta', 'ru', 'vi']
langs = ['fa', 'hi', 'es', 'it', 'da', 'he', 'sv', 'bg', 'lv', 'hu', 'tr', 'ta', 'ru', 'vi']
# langs = ['en', 'cs']
base_format = "logs_token_exp_sign/log-{}-10k-noseq-pginit-{}char-05dr/testout.txt"

with open("logs_token_exp_10k-sign.txt","w","utf-8") as outfile:
    for lg in langs:

        # mchar vs. nochar
        nofile = open(base_format.format(lg, "no"), "r", "utf-8")
        mfile = open(base_format.format(lg, "m"), "r", "utf-8")
        wn, wm, _, wilwo, wno, wmo, wilwov = extract_stats(nofile, mfile)
        print lg, "w/o", wn, wm, wilwo, wno, wmo, wilwov

        # bothchar vs. tagchar
        tagfile = open(base_format.format(lg, "tag"), "r", "utf-8")
        bothfile = open(base_format.format(lg, "both"), "r", "utf-8")
        wt, wb, _, wilw, wto, wbo, wilwv = extract_stats(tagfile, bothfile)
        print lg, "with", wt, wb, wilw, wto, wbo, wilwv

        outfile.write("\t".join([lg, wilwo < 0.05, wilw < 0.05, wilwov < 0.05, wilwv < 0.05]) + "\n")
