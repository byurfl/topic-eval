import src.algorithms.sam.util as util
from src.algorithms.sam.reader import Reader
import numpy as np


class SAM:
    def __init__(self, corpus, stopwords=None, topics=10):

        # when stopping criteria hasn't changed by more than
        # epsilon for a few iterations, stop topic discovery
        self.CONVERGENCE = False
        self.EPSILON = 0.001
        self.LAMBDA = 0.0001

        self.reader = Reader(stopwords)
        self.reader.read_corpus(corpus)

        self.vocabulary = self.reader.vocabulary
        self.vocab_size = len(self.vocabulary)
        self.documents = self.reader.documents
        self.num_docs = self.documents.shape[0]

        # initialize model hyperparameters
        self.num_topics = topics

        self.xi = 500.0
        self.m = util.l2_normalize(np.random.rand(self.vocab_size))
        self.alpha = np.random.rand(self.num_topics, self.num_docs)
        self.k0 = 50.0
        self.k = 500.0

        # initialize variational parameters
        self.vAlpha = np.random.rand(self.num_topics, self.num_docs)
        self.vMu = util.l2_normalize(np.random.rand(self.vocab_size, self.num_topics))
        self.vM = util.l2_normalize(np.random.rand(self.vocab_size))

    def update_model_params(self):

        # self.xi =
        self.m = util.l2_normalize(np.sum(self.vMu, axis=1))
        # self.alpha =

    def update_free_params(self):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        A_V_k0 = util.bessel_approx(self.vocab_size, self.k0)
        topic_mean_sum = np.sum(self.vMu)

        # self.vAlpha =

        # self.vMu =

        self.vM = util.l2_normalize(self.k0 * A_V_k0 * self.m +
                                    A_V_xi * A_V_k0 * self.xi *
                                    topic_mean_sum + 2 * self.LAMBDA * self.vM)

    def do_EM(self):
        old_result = 0
        while not self.CONVERGENCE:
            self.do_E()
            new_result = self.do_M()
            if new_result - old_result < self.EPSILON:
                self.CONVERGENCE = True

    def do_E(self):
        self.update_free_params()
        self.update_model_params()

    def do_M(self):
        return 0
        
    def update_xi(self):

    
    def update_alpha(self):
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', required=True, help='Path to the location of your corpus. ' +
                                                              'This can be either a directory or a single file.')
    parser.add_argument('-s', '--stopwords', help='Optional path to a file containing stopwords.')
    args = parser.parse_args()

    if args.stopwords:
        model = SAM(args.corpus)
    else:
        model = SAM(args.corpus, args.stopwords)

    model.do_EM()
