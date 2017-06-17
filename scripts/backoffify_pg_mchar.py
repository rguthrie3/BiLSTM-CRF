from __future__ import division
import cPickle
import argparse
import codecs
import numpy as np

def read_text_embs(filename):
    words = []
    embs = []
    with codecs.open(filename, "r", "utf-8") as f:
        for line in f:
            split = line.split()
            if len(split) > 2:
                words.append(split[0])
                embs.append(np.array([float(s) for s in split[1:]]))
    return words, embs

def output_word_vector(word, embed, outfile):
    outfile.write(word + " ")
    for i in embed:
        outfile.write(str(i) + " ")
    outfile.write("\n")

parser = argparse.ArgumentParser()
parser.add_argument("--vectors", dest="vectors", help="Pickle file from which to get word vectors")
parser.add_argument("--mchar-file", dest="mchar_file", help="File containing embeddings with mchar but no lowercase backoff")
parser.add_argument("--output", dest="output", help="Output location")
options = parser.parse_args()

# Read in the existing, mchar-trained vectors
orig_words, orig_embs = read_text_embs(options.mchar_file)

# Replace needed with lowercased vecs
words, embs = cPickle.load(open(options.vectors, "r"))
word_to_ix = {w : i for (i,w) in enumerate(words)}
with codecs.open(options.output, "w", "utf-8") as outfile:
    changed = 0
    total = len(orig_words)
    for orig_word, orig_emb in zip(orig_words, orig_embs):
        if orig_word in words: # taken from original pg file, all is good
            output_word_vector(orig_word, orig_emb, outfile)
        elif orig_word.lower() in words: # take lower - THIS IS THE CHANGE
            embed = embs[word_to_ix[orig_word.lower()]]
            output_word_vector(orig_word, embed, outfile)
            changed += 1            
        else: # taken from matching char, also good
            output_word_vector(orig_word, orig_emb, outfile)
    print "Total Number of output words:", total
    print "Changed:", changed
    print "Percentage changed:", changed / total
