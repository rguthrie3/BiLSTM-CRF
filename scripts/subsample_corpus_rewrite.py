from random import randint
from collections import Counter, deque
import argparse
import codecs
import nltk.tokenize
import re


def doc_splitter(data):
    line = data.readline()
    ret = line
    while line != "\n" and line != "":
        line = data.readline()
        ret += line
    if line == "":
        return None # EOF
    while line == "\n":
        line = data.readline()
    if line == "":
        return None # EOF
    data.readline()
    return ret


class SequenceGenerator:


    def __init__(self, filename, seq_len, document_splitter, tokenizer):
        self.seq_len = seq_len
        self.file = codecs.open(filename, "r", "utf-8", "utf-8")
        self.buffer = deque()
        self.line = None
        self.done = False
        self.document_splitter = document_splitter
        self.tokenizer = tokenizer


    def __iter__(self):
        return self


    def next(self):
        if self.done:
            raise StopIteration

        # Check if we have any leftover sequences from the last document
        # we processed
        if len(self.buffer) > 0:
            return self.buffer.popleft()
        
        doc = self.document_splitter(self.file)
        if doc is None:
            self.done = True
            raise StopIteration
        doc = self._tokenize(doc)

        # If a document is too short to contain a full sequence, just
        # ignore that document
        while len(doc) < self.seq_len:
            doc = self.document_splitter(self.file)
            if doc is None:
                self.done = True
                raise StopIteration
            doc = self._tokenize(doc)

        seq = doc[:self.seq_len]

        # Check if there are any more sequences in this document, which will
        # be pushed into the buffer and returned as the next sequences
        if len(doc) >= 2 * self.seq_len: # there is another full sequence in this doc
            for i in range(self.seq_len, len(doc), self.seq_len):
                if len(doc) > i + self.seq_len:
                    self.buffer.append(doc[i:i + self.seq_len])

        return seq


    def _tokenize(self, string):
        if self.tokenizer is not None:
            return self.tokenizer(string)
        else:
            return string.split()
            


def res_sample(gen, k):
    output = [None] * k
    for i in xrange(k):
        output[i] = gen.next()
    counter = 0
    for thing in gen:
        r = randint(0, counter + k)
        if r < k:
            output[r] = thing
        counter += 1
        print counter
    return output


def sample(gen):
    output = []
    for sample in gen:
        output.append(sample)
    return output

def preprocess(x):
    if re.match("-?[0-9]*[\.,]?[0-9]+", x) is not None:
        return "NUM"
    elif re.match("(\/*www\.)|(\/*\.com)", x) is not None:
        return "URL"
    else:
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs=1, help="Text file to process")
    parser.add_argument("-o", dest="output", required=True, help="Output filename")
    parser.add_argument("--num-types", dest="num_types", default=50000, help="Number of unique types to preserve if doing preprocessing")
    parser.add_argument("--no-subsample", dest="do_subsample", default=True, action="store_false", help="Whether or not to subsample the text")
    parser.add_argument("--subsample-token-count", required=True, dest="token_count", help="Total number of tokens that should be in the output")
    parser.add_argument("--subsample-seq-len", dest="seq_len", default=50, help="Length of the chunks that will be sampled")
    parser.add_argument("--subsample-dont-split-on-token", required=False, dest="dont_split_on_token", help="Don't output a sequence that crosses a certain token (e.g ==END-OF-DOCUMENT==")
    parser.add_argument("--language", required=True, dest="language", help="NLTK tokenizer language")
    parser.add_argument("--vocab", dest="vocab", help="vocabulary output file")
    options = parser.parse_args()

    # Subsample the data
    num_samples = int(options.token_count) / int(options.seq_len)
    seq_generator = SequenceGenerator(options.input[0], int(options.seq_len), doc_splitter, None)

    if options.do_subsample:
        sampled_data = res_sample(seq_generator, num_samples)
        print 'Done sampling'
    else:
        sampled_data = sample(seq_generator)
        print 'Done pseudo-sampling'

    # Preprocess the data
    word_counts = Counter()
    for sequence in sampled_data:
        word_counts.update(sequence)

    n_most_common_words = set(sorted(word_counts.keys(), key=lambda x: word_counts[x])[-int(options.num_types):])
    with codecs.open(options.output, "w", "utf-8") as output_file:
        for sequence in sampled_data:
            for word in sequence:
                word = preprocess(word)
                if word in n_most_common_words or word == "NUM":
                    output_file.write(" " + word)
                else:
                    output_file.write(" UNK")
            output_file.write("\n")
            
    if options.vocab is not None:
        with codecs.open(options.vocab, "w", "utf-8") as vocab_file:
            for w in n_most_common_words:
                vocab_file.write(w + "\n")


if __name__ == "__main__":
    main()

