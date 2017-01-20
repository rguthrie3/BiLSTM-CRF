from __future__ import division
import cPickle
import argparse
import morfessor
import codecs
import numpy as np

def output_word_vector(word, embed, outfile):
    outfile.write(word + " ")
    for i in embed:
        outfile.write(str(i) + " ")
    outfile.write("\n")

parser = argparse.ArgumentParser()
parser.add_argument("--vectors", dest="vectors", help="Pickle file from which to get word vectors")
parser.add_argument("--vocab", dest="vocab", help="File containing words to output embeddings for")
parser.add_argument("--morf", dest="morfessor_model", help="Trained morfessor model")
parser.add_argument("--output", dest="output", default="word_and_morpho_embeds.txt", help="Output location")
parser.add_argument("--lowercase-backoff", dest="lowercase_backoff", action="store_true", help="Use lowercased segmentation if not available for capitalized word")
parser.add_argument("--in-vocab-only", dest="in_vocab_only", action="store_true", help="Only output an embedding if it is in-vocab")
parser.add_argument("--morpho-only", dest="morpho_only", action="store_true", help="Only use the morpheme embeddings")
parser.add_argument("--sum-embed", dest="sum_embed", action="store_true", help="Flag to generate embeddings for SumEmbed model")
parser.add_argument("--polyglot", dest="polyglot", action="store_true", help="Model is Polyglot")
options = parser.parse_args()

# Read in the output vocab
with codecs.open(options.vocab, "r", "utf-8") as f:
    output_words = set([ line.strip() for line in f ])

# Polyglot is easy
if options.polyglot:
    words, embs = cPickle.load(open(options.vectors, "r"))
    word_to_ix = {w : i for (i,w) in enumerate(words)}
    with codecs.open("word_and_morpho_embeds.txt", "w", "utf-8") as outfile:
        in_vocab = 0
        total = len(output_words)
        for orig_word in output_words:
            if orig_word not in words and options.lowercase_backoff:
                word = word.lower()
            else:
                word = orig_word
            if word in words:
                embed = embs[word_to_ix[word]]
                output_word_vector(orig_word, embed, outfile)
                in_vocab += 1
        print "Total Number of output words:", total
        print "Total in Training Vocabulary:", in_vocab
        print "Percentage in-vocab:", in_vocab / total
    exit()

# Read in model
D = cPickle.load(open(options.vectors, "r"))
word_to_ix = D["word_to_ix"]
morpho_to_ix = D["morpho_to_ix"]
word_embeddings = D["word_embeddings"]
morpho_embeddings = D["morpheme_embeddings"]
if options.morfessor_model is not None:
    morfessor_model = morfessor.MorfessorIO().read_binary_model_file(options.morfessor_model)

# If we get an unknown morpheme, just use 0
np.append(morpho_embeddings, np.zeros((1,morpho_embeddings.shape[1])), axis=0)


with codecs.open(options.output, "w", "utf-8") as outfile:
    in_vocab = 0
    out_vocab = 0
    total = len(output_words)
    if options.morpho_only:
        morphologically_complex_words_count = 0
        have_atleast_1_morpho_count = 0
        have_all_morphos_count = 0
        for word in output_words:
            if options.in_vocab_only and word not in word_to_ix: continue
            morpheme_indices = [morpho_to_ix.get(m, -1) for m in morfessor_model.viterbi_segment(word)[0]]
            have_atleast_1_morpho = reduce(lambda x,y: y != -1 or x, morpheme_indices, False)
            have_all_morphos = reduce(lambda x,y: y != -1 and x, morpheme_indices, True)
            if have_atleast_1_morpho:
                have_atleast_1_morpho_count += 1
            if have_all_morphos:
                have_all_morphos_count += 1
            if len(morpheme_indices) > 1:
                morphologically_complex_words_count += 1
            embed = np.array([morpho_embeddings[i] for i in morpheme_indices]).sum(axis=0)
            output_word_vector(word, embed, outfile)
        print "Total Number of words:", total
        print "Total Number of Morphologically Complex Words (num morphemes > 1):", morphologically_complex_words_count
        print "Total Number of Words for which we have at least 1 morpheme embedding:", have_atleast_1_morpho_count
        print "Total Number of Words for which we have all morpheme embeddings:", have_all_morphos_count
    else:
        for orig_word in output_words:
            if orig_word not in word_to_ix and options.lowercase_backoff:
                word = word.lower()
            else:
                word = orig_word
            if word in word_to_ix:
                embed = word_embeddings[word_to_ix[word]]
                if options.sum_embed: # For SumEmbed, we include the morpheme embeddings in the sum even for in-vocab words
                    morpho_embed = np.array([ morpho_embeddings[morpho_to_ix.get(m, -1)] for m in morfessor_model.viterbi_segment(word)[0] ]).sum(axis=0)
                    embed = embed + morpho_embed
                output_word_vector(orig_word, embed, outfile)
                in_vocab += 1
            elif not options.in_vocab_only:
                embed = np.array([ morpho_embeddings[morpho_to_ix.get(m, -1)] for m in morfessor_model.viterbi_segment(word)[0] ]).sum(axis=0)
                output_word_vector(orig_word, embed, outfile)
                out_vocab += 1	
        print "Total Number of wordvectors.org words:", total
        print "Total in Training Vocabulary:", in_vocab
        print "Total out of Training Vocabulary", out_vocab
        print "Percentage in-vocab:", in_vocab / total
