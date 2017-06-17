from __future__ import division
import codecs
import cPickle
import collections

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

langs = "en it da lv tr vi hu ta fa ru sv he bg hi cs es".split()

for lg in langs:
    # get polyglot vocab
    with open("data/embs/pg/{}-polyglot.pkl".format(lg),"r") as pg_file:
        pg_words_list, _ = cPickle.load(pg_file)
        pg_words = set(pg_words_list)
        #print "{}: found {} Polyglot words".format(lg, len(pg_words))
    
    # get oov word list
    with codecs.open("vocabs/{}-vocab.txt".format(lg),"r", "utf-8") as ud_vocab_file:
        ud_vocab_words = set([w.strip() for w in ud_vocab_file.readlines()])
        #print "{}: found {} UD words".format(lg, len(ud_vocab_words))
    
    # get oovs
    oovs = ud_vocab_words - pg_words
    oov_typ_rate = len(oovs) * 100 / len(ud_vocab_words)
    #print "{}: OOV types - {}".format(lg, len(oovs), oovs)
    
    # count oovs in UD dataset
    with open("datasets/{}_mtags-pos.pkl".format(lg),"r") as udds_file:
        udds = cPickle.load(udds_file)
        oov_idxs = set([i for w,i in udds["w2i"].iteritems() if w in oovs])
        assert len(oov_idxs) == len(oovs)
        #print "{}: OOV types in DS - {}".format(lg, len(oov_idxs))
        total_toks = 0
        oov_toks = 0
        for inst in udds["training_instances"] + udds["dev_instances"] + udds["test_instances"]:
            total_toks += len(inst.sentence)
            for word in inst.sentence:
                if word in oov_idxs:
                    oov_toks += 1
    
    oov_tok_rate = oov_toks * 100 / total_toks
    print lg, len(ud_vocab_words), len(oovs), oov_typ_rate, oov_tok_rate
    #print "{}: found {:.2f}% OOV tokens ({} of {})".format(lg, oov_tok_rate, oov_toks, total_toks)
    