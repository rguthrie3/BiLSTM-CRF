lang = 'ta' # he, en

pg_words, _ = cPickle.load(open("data/embs/pg/{}-polyglot.pkl".format(lang),"r"))
ud_vocab_words = set([w.strip() for w in codecs.open("vocabs/{}-vocab.txt".format(lang),"r","utf-8").readlines()])
oovs = ud_vocab_words - set(pg_words)
with codecs.open("vocabs/{}-ud-pg-alpha-oovs.txt".format(lang),"w","utf-8") as file:
    file.write("\n".join([w for w in oovs if w.isalpha()]))
