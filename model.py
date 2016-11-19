import collections
import argparse
import random
import dynet as dy
import numpy as np

Instance = collections.namedtuple("Instance", ["sentence", "tags"])


class BiLSTM_CRF:

    def __init__(self, vocab_size, tagset_size, num_lstm_layers, embedding_dim, hidden_dim):
        self.model       = dy.Model()
        self.tagset_size = tagset_size

        # Word embedding parameters
        self.words_lookup = self.model.add_lookup_parameters((vocab_size, embedding_dim))

        # LSTM parameters
        self.bi_lstm = dy.BiRNNBuilder(num_lstm_layers, embedding_dim, hidden_dim, self.model, dy.LSTMBuilder)
        
        # Matrix that maps from Bi-LSTM output to num tags
        self.lstm_to_tags_params = self.model.add_parameters((tagset_size, hidden_dim))

        # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
        self.transitions = self.model.add_lookup_parameters((tagset_size, tagset_size))


    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        embeddings = [self.words_lookup[w] for w in sentence]
        lstm_out   = self.bi_lstm.transduce(embeddings)

        H = dy.parameter(self.lstm_to_tags_params)
        probs = []
        for rep in lstm_out:
            prob_t = dy.log_softmax(H * rep)
            probs.append(prob_t)

        return probs


    def score_sentence(self, observations, tags):
        assert len(observations) == len(tags)
        init_vvars = [-1e10] * self.tagset_size
        init_vvars[-2] = 0 # <Start> has all the probability
        score      = dy.scalarInput(0)
        tags = [self.tagset_size - 2] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(self.transitions[tags[i+1]], tags[i]) + obs[tags[i+1]]
        score = score + dy.pick(self.transitions[self.tagset_size - 1], tags[-1])
        return score


    def viterbi_loss(self, sentence, tags):
        observations = self.build_tagging_graph(sentence)
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations, tags)
        if viterbi_tags != tags:
            gold_score = self.score_sentence(observations, tags)
            return viterbi_score - gold_score, viterbi_tags
        else:
            return dy.scalarInput(0), viterbi_tags


    def neg_log_loss(self, sentence, tags):
        observations  = self.build_tagging_graph(sentence)
        forward_score = self.forward(observations)
        gold_score    = self.score_sentence(observations, tags)
        return -(gold_score - forward_score)


    def forward(self, observations):

        def log_sum_exp(scores):
            scores_np            = scores.npvalue()
            max_score            = np.max(scores_np, axis=0)
            max_score_broadcast  = max_score.repeat(self.tagset_size).reshape(self.tagset_size, self.tagset_size).T
            print max_score
            max_score_expr       = dy.inputVector(max_score)
            max_score_bcast_expr = dy.inputMatrix(max_score_broadcast.flatten(), (self.tagset_size, self.tagset_size))
            return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_bcast_expr))))

        init_alphas     = [-1e10] * self.tagset_size
        init_alphas[-2] = 0
        for_expr        = dy.inputVector(init_alphas)
        trans_matrix    = dy.concatenate_cols([self.transitions[idx] for idx in xrange(self.tagset_size)])
        for i, obs in enumerate(observations):
            obs_matrix  = dy.transpose(dy.concatenate_cols([obs] * self.tagset_size))
            prev_matrix = dy.concatenate_cols([for_expr] * self.tagset_size)
            scores      = obs_matrix + prev_matrix + trans_matrix
            for_expr    = log_sum_exp(scores)
        terminal_expr = for_expr + self.transitions[self.tagset_size - 1]
        alpha         = dy.log(dy.sum_cols(dy.exp(dy.transpose(terminal_expr))))
        return alpha


    def viterbi_decoding(self, observations):
        backpointers = []
        init_vvars   = [-1e10] * self.tagset_size
        init_vvars[-2] = 0 # <Start> has all the probability
        for_expr     = dy.inputVector(init_vvars)
        trans_exprs  = [self.transitions[idx] for idx in range(self.tagset_size)]
        for i, obs in enumerate(observations):
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id  = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(next_tag_expr)
            for_expr = dy.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[-1]
        terminal_arr  = terminal_expr.npvalue()
        best_tag_id   = np.argmax(terminal_arr)
        path_score    = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id] # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        best_path.pop() # Remove the start symbol
        # Return best path and best path's score
        return best_path, path_score

    @property
    def model(self):
        return self.model


def read(filename, w2i, t2i):
    instances = []
    with open(filename, "r") as f:
        sentence = []
        tags     = []
        for line in f:
            line = line.rstrip().lstrip()
            if line == "===":
                instances.append(Instance(sentence, tags))
                sentence = []
                tags     = []
            else:
                split     = line.split("###")
                word, tag = split
                if word not in w2i:
                    w2i[word] = len(w2i)
                if tag not in t2i:
                    t2i[tag] = len(t2i)
                sentence.append(w2i[word])
                tags.append(t2i[tag])
    return instances

parser = argparse.ArgumentParser()
parser.add_argument("--num-epochs", default=15, dest="num_epochs", help="Number of full passes through training set")
parser.add_argument("--lstm-layers", default=2, dest="lstm_layers", help="Number of LSTM layers")
parser.add_argument("--embedding-dim", default=128, dest="embedding_dim", help="Size of Word embeddings")
parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", help="Size of LSTM hidden layers")
parser.add_argument("--learning-rate", default=0.001, dest="learning_rate", help="Initial learning rate")
parser.add_argument("--dropout", default=-1, dest="dropout", help="Amount of dropout to apply to LSTM part of graph")
options = parser.parse_args()

w2i = {}
t2i = {}
trn_instances = read("test.txt", w2i, t2i) 

bilstm_crf = BiLSTM_CRF(len(w2i), len(t2i), options.lstm_layers, options.embedding_dim, options.hidden_dim)
trainer    = dy.AdamTrainer(bilstm_crf.model)

for epoch in xrange(options.num_epochs):
    random.shuffle(trn_instances)
    train_loss = 0.0
    for instance in trn_instances:
        loss_expr   = bilstm_crf.neg_log_loss(instance.sentence, instance.tags)
        loss        = loss_expr.scalar_value()
        train_loss += loss
        loss_expr.backward()
        trainer.update()
