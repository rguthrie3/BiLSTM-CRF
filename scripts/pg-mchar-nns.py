from __future__ import division
import numpy as np
import cPickle
import codecs
import collections

NN_COUNT = 10

# lang, inter, eps, inter2 = 'en', '-', '100', ''
lang, inter, eps, inter2 = 'es', '', '60', 's'
# lang, inter, eps, inter2 = 'he', '', '60', 's'
# lang, inter, eps, inter2 = 'ta', '', '60', 's'

def dist(vec1, vec2):
    return 1.0 - (vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))) # cosine
    #return np.linalg.norm(vec1 - vec2) # euclidean

# get polyglot vocab
with open("data/embs/pg/{}-polyglot.pkl".format(lang), "r") as pg_file:
    pg_words, pg_embs = cPickle.load(pg_file)

pg_all = zip(pg_words, pg_embs)

# get oovs
with codecs.open("vocabs/{}-ud-pg-alpha-oovs.txt".format(lang), "r", "utf-8") as ud_vocab_file:
    oovs = [w.strip() for w in ud_vocab_file.readlines()]# if w.strip().lower() not in pg_words]
    print len(oovs)

# get oov vecs
oov_vecs = {}
with codecs.open("polyglot_trainer/{}-cpg{}fb-{}ep{}-embs.txt".format(lang, inter, eps, inter2), "r", "utf-8") as ud_embs_file:
    for l in ud_embs_file.readlines():
        parts = l.split()
        if len(parts) > 2 and parts[0] in oovs:
            oov_vecs[parts[0]] = np.array([float(s) for s in parts[1:]])

assert len(oov_vecs) == len(oovs)

# find NNs, print
with codecs.open("{}-udoov-pg-nns-cosine.txt".format(lang), "w", "utf-8") as out:
#with codecs.open("{}-udoov-pg-nns-euc.txt".format(lang), "w", "utf-8") as out:
    for w in oovs:
        vec = oov_vecs[w]
        nns = [x[0] for x in sorted(pg_all, key=lambda x: dist(vec, x[1]))][:NN_COUNT]
        out.write(w + "\t" + "\t".join(nns) + "\n")
    
