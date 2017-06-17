'''
finds the rate of words in training corpus tagged with any kind of morphological tag.
'''
from __future__ import division
all_langs = ['ta', 'fa', 'ru', 'sv', 'he', 'bg', 'hi', 'cs', 'es', 'en', 'da', 'it', 'tr', 'lv', 'vi', 'hu']
for l in all_langs:
    if l == 'vi': continue
    total_len = 0
    tagged_words = 0
    with open("datasets/{}_mtags-dd.pkl".format(l),'r') as udds_f:
        udds = cPickle.load(udds_f)
        l_t2is = udds['t2is']
        tr_insts = udds['training_instances']
        for ins in tr_insts:
            for i in xrange(len(ins)):
                 total_len += 1
                 if len([a for a,vs in ins.tags.items() if len(vs) > i and a != 'POS' and vs[i] != l_t2is[a]['<NONE>']]) > 0: tagged_words += 1
	print l, total_len, tagged_words, tagged_words / total_len