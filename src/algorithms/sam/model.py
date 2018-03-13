
# Anchor words needs a home!
import os

TOP = 15
BOTTOM = 15
VERBOSE = True
ITERATIONS = 10
OUR_READER = True
LIMIT_VOCAB = True

# Vocab pruning
PRUNE_VOCAB = False
PERC_VOCAB_TO_KEEP = .5
START_PRUNE_ITERATION = 3

if os.environ["COMPUTERNAME"] == 'DALAILAMA':
    import sys
    PRUNE_VOCAB = True
    OUR_READER = True
    ITERATIONS = 30
    LIMIT_VOCAB = False
    path = r"D:\PyCharm Projects\py-sam-master\topic-eval"
    os.environ["HOME"] = r"D:\PyCharm Projects\py-sam-master\topic-eval\data\corpus;"
    #print(os.getenv("HOME"))
    if True:
        VERBOSE = False
        TOP = 3
        BOTTOM = 3
    sys.path.append(path)
    os.chdir(path)

import src.algorithms.sam.util as util
import numpy as np
from scipy.special import gammaln, psi, polygamma

class SAM:
    def __init__(self, corpus, topics, stopwords=None, log_file=None, corpus_encoding = 'utf-8'):
        self.corpus = corpus
        if log_file == None:
            self.log_file = corpus + '_log.txt'
        else:
            self.log_file = log_file

        self.loss_file = corpus + '_loss.csv'

        if os.path.isdir(self.log_file):
            self.log_file = os.path.join(self.log_file, 'log.txt')

        # truncate log file if it exists, create it if it doesn't
        with open(self.log_file, mode='w', encoding='utf-8'):
            pass

        if OUR_READER:
            from src.algorithms.sam.reader import Reader
            self.reader = Reader(stopwords, corpus_encoding=corpus_encoding, use_vocab_dict = LIMIT_VOCAB)
            self.reader.read_corpus(corpus)

            self.vocabulary = self.reader.vocabulary
            self.vocab_size = self.reader.vocab_size

            self.sorted_terms = np.asarray(['' for _ in range(self.vocab_size)], dtype=object)
            for key, value in self.reader.terms_to_indices.items():
                self.sorted_terms[value] = key

            with open(self.log_file, mode='w', encoding='utf-8') as log_out:
                 for term in self.sorted_terms:
                     log_out.write(term + '\n')

            self.documents = self.reader.documents
            self.num_docs = self.documents.shape[1]
        else:
            from src.algorithms.sam.corpus.corpus import CorpusReader
            self.reader = CorpusReader(corpus, data_series='sam')
            self.vocab_size = self.reader.dim
            self.num_docs = self.reader.num_docs
            self.documents = np.empty((self.vocab_size, self.num_docs))
            for d in range(self.reader.num_docs):
                self.documents[:,d] = self.reader.read_doc(d).T

            self.sorted_terms = np.array([line.strip() for line in open(self.reader.filename + '.wordlist')], str)

        # initialize model hyperparameters
        self.num_topics = topics

        self.xi = 5000.0
        #self.m = util.l2_normalize(np.random.rand(self.vocab_size))
        self.m = util.l2_normalize(np.ones(self.vocab_size))  # Parameter to p(mu)

        #self.alpha = np.random.rand(self.num_topics)
        self.alpha = np.ones(self.num_topics) * 1.0 + 1.0

        self.k0 = 10.0
        self.k = 5000.0

        # initialize variational parameters
        self.vMu = util.l2_normalize(np.random.rand(self.vocab_size, self.num_topics))
        self.vM = util.l2_normalize(np.random.rand(self.vocab_size))

        # self.vAlpha = np.random.rand(self.num_topics, self.num_docs)
        self.vAlpha = np.empty((self.num_topics, self.num_docs))
        for d in range(self.num_docs):
            distances_from_topics = np.abs(util.cosine_similarity(self.documents[:, d], self.vMu)) + 0.01
            self.vAlpha[:, d] = distances_from_topics / sum(distances_from_topics) * 3.0

        self.reset_loss_history()
        # For pruning
        if PRUNE_VOCAB:
            self.backup_vocab()
            self.deleted_words = []
            pass

    def reset_loss_history(self):
        # Record vectors
        self.loss_updates = {"xi":[],"m":[],"alpha":[],"vMu":[],"vAlpha":[],"vM":[]}
        self.loss_updates["vM"].append(self.vM)
        self.loss_updates["m"].append(self.m)

    def __eq__(self, other):
        try:
            if object.__eq__(self, other):
                return True
            if self.vocab_size != other.vocab_size:
                return False
            if self.vocabulary != other.vocabulary:
                return False
            if self.num_docs != other.num_docs:
                return False
            if self.documents != other.documents:
                return False
            if self.num_topics != other.num_topics:
                return False
            if self.xi != other.xi:
                return False
            if self.m != other.m:
                return False
            if self.alpha != other.alpha:
                return False
            if self.k0 != other.k0:
                return False
            if self.k != other.k:
                return False
            if self.vAlpha != other.vAlpha:
                return False
            if self.vMu != other.vMu:
                return False
            if self.vM != other.vM:
                return False
            return True
        except:
            return False

    def reset(self):
        self.reset_vocab()
        self.reset_loss_history()

    def backup_vocab(self):
        self.vocab_size_b = self.vocab_size
        self.vMu_b =  self.vMu[:,:]
        self.vM_b  =  self.vM[:]
        self.m_b =  self.m[:]
        self.documents_b = self.documents[:,:]

    def reset_vocab(self):
        self.vocab_size  = self.vocab_size_b
        self.vMu =  self.vMu_b[:,:]
        self.vM  =  self.vM_b[:]
        self.m =  self.m_b[:]
        self.documents = self.documents_b[:,:]

    def delete_vocab_words(self, row_indices):
        self.vocab_size -= len(row_indices)
        #print(len(row_indices))
        #print(self.vMu)
        #print(row_indices)
        self.vMu =  np.delete(self.vMu, row_indices, axis=0)
        #print(self.vMu)
        import time
        #time.sleep(14)

        self.vM  =  np.delete(self.vM, row_indices, axis=0)
        self.m =  np.delete(self.m, row_indices, axis=0)
        self.documents = np.delete(self.documents, row_indices, axis=0)

    def prune_topics(self, limit_by = 10):
        # n = number of topics to remove

        self.vocab_size -= limit_by

        #self.vMu = self.vMu[:self.vocab_size,:]
        #self.vM = self.vM[:self.vocab_size]
        #self.m = self.m[:self.vocab_size]
        #self.documents = self.documents[:self.vocab_size,:]

        # Delete words that are all ~0?


        # Calculate topic variance
        #mean_adj = self.vMu/self.vMu.mean(axis=1, keepdims = True)
        #var = mean_adj.var(axis=1)

        var = self.vMu.var(axis=1)
        delete_list = var.argsort()[:limit_by][::-1]
        self.delete_vocab_words(delete_list)
        self.deleted_words.append(delete_list)

    def update_model_params(self):
        self.m = util.l2_normalize(np.sum(self.vMu, axis=1))
        self.loss_updates["m"].append(self.m)

        self.update_xi()
        self.update_alpha()


    def vAlpha_likelihood(self):
        alpha0 = np.sum(self.alpha)
        psi_vAlpha = psi(self.vAlpha)
        vAlpha0s = np.sum(self.vAlpha, axis=0)
        psi_vAlpha0s = psi(vAlpha0s)

        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        sum_rhos = sum(self.calc_rhos())

        likelihood = np.dot(util.make_row_vector(self.alpha - 1.0), psi_vAlpha).sum() \
                    - (alpha0 - self.num_topics) * psi_vAlpha0s.sum() \
                    + self.num_docs * gammaln(alpha0) \
                    - self.num_docs * gammaln(self.alpha).sum() \
                    + self.k * sum_rhos \
                    - np.sum((self.vAlpha - 1.0) * psi_vAlpha) \
                    + np.sum(psi_vAlpha0s * (vAlpha0s - self.num_topics)) \
                    - np.sum(gammaln(vAlpha0s)) \
                    + np.sum(gammaln(self.vAlpha))

        return likelihood

    def expected_squared_norms(self):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        vAlpha0s = np.sum(self.vAlpha, axis=0)
        vAlphas_squared = np.sum(self.vAlpha ** 2, axis=0)
        A_V_xi_squared = A_V_xi ** 2

        vMu_squared = np.dot(self.vMu.T, self.vMu)
        vMu_vAlpha_squared_1 = np.dot(self.vAlpha.T, vMu_squared)
        vMu_vAlpha_squared_2 = vMu_vAlpha_squared_1.T * self.vAlpha
        vMu_vAlpha_squared = np.sum(vMu_vAlpha_squared_2, axis=0)

        result = (vAlpha0s + (1.0 - A_V_xi_squared) * vAlphas_squared + A_V_xi_squared * vMu_vAlpha_squared) / \
                 (vAlpha0s * (vAlpha0s + 1.0))

        return result

    def expected_square_norms_gradient(self):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        vAlpha0s = np.sum(self.vAlpha, axis=0)
        square_norms = self.expected_squared_norms()

        A_V_xi_squared = A_V_xi ** 2
        vMu_vAlpha_vMu = np.dot(self.vAlpha.T, np.dot(self.vMu.T, self.vMu))
        doc_weights = 1.0 / (vAlpha0s * (vAlpha0s + 1.0))
        gradient = 1 + 2 * (1-A_V_xi_squared) * self.vAlpha + 2 * A_V_xi_squared * vMu_vAlpha_vMu.T
        gradient = gradient - square_norms * util.make_row_vector(2 * vAlpha0s + 1.0)
        gradient = gradient * util.make_row_vector(doc_weights)

        return gradient

    def calc_rhos(self):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        expecteds = self.expected_squared_norms()
        vAlpha0s = np.sum(self.vAlpha, axis=0)
        vMu_docs = self.vMu.T.dot(self.documents)

        result_1 = util.make_row_vector(1.0 / vAlpha0s / np.sqrt(expecteds))
        result_2 = self.vAlpha * result_1
        result_3 = result_2 * vMu_docs
        result_4 = np.sum(result_3, axis=0)
        result = result_4 * A_V_xi

        return result

    def rho_vAlpha_gradient(self):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        vAlpha0s = np.sum(self.vAlpha, axis=0)

        expected_square_norms = self.expected_squared_norms()
        square_norms_gradient = self.expected_square_norms_gradient()

        vMu_docs = np.dot(self.vMu.T, self.documents)
        vMu_vAlpha_docs = np.sum(self.vAlpha * vMu_docs, axis=0)

        gradient = vMu_docs / util.make_row_vector(vAlpha0s)
        gradient = gradient - util.make_row_vector(vMu_vAlpha_docs / vAlpha0s ** 2)
        gradient = gradient / util.make_row_vector(np.sqrt(expected_square_norms))

        S_d = vMu_vAlpha_docs / vAlpha0s / (2 * expected_square_norms ** (3.0/2.0))
        gradient = A_V_xi * (gradient - square_norms_gradient * util.make_row_vector(S_d))

        return gradient

    def vAlpha_gradient(self):
        alpha0 = np.sum(self.alpha)
        vAlpha0s = np.sum(self.vAlpha, axis=0)
        psi_vAlpha_gradient = polygamma(1, self.vAlpha)
        psi_vAlpha0s_gradient = polygamma(1, vAlpha0s)

        rho_gradient = self.rho_vAlpha_gradient()

        gradient = util.make_col_vector(self.alpha - 1.0) * psi_vAlpha_gradient \
                    + self.k * rho_gradient - (self.vAlpha - 1.0) * psi_vAlpha_gradient

        row_constant = -psi_vAlpha0s_gradient * (alpha0 - self.num_topics) + \
                       psi_vAlpha0s_gradient * (vAlpha0s - self.num_topics)

        gradient = gradient + util.make_row_vector(row_constant)

        return gradient

    def vMu_likelihood(self):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        A_V_k0 = util.bessel_approx(self.vocab_size, self.k0)
        vM_dot_vMu = np.dot(self.vM.T, np.sum(self.vMu, axis=1))
        sum_rhos = sum(self.calc_rhos())

        return (A_V_xi * A_V_k0 * self.xi * vM_dot_vMu) + (self.k * sum_rhos)

    def vMu_gradient(self):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        A_V_k0 = util.bessel_approx(self.vocab_size, self.k0)
        A_V_xi_squared = A_V_xi ** 2
        squared_norms = self.expected_squared_norms()
        vAlpha0s = np.sum(self.vAlpha, axis=0)

        first_part = np.dot(self.documents, (self.vAlpha * A_V_xi / util.make_row_vector(vAlpha0s * np.sqrt(squared_norms))).T)
        doc_weights = A_V_xi / vAlpha0s / (2 * squared_norms ** (3.0/2.0)) * (self.vAlpha * np.dot(self.vMu.T, self.documents)).sum(axis=0).T

        second_doc_weights = 2*(1-A_V_xi_squared) / (vAlpha0s*(vAlpha0s+1.0))
        second_part = np.sum(doc_weights * second_doc_weights) * self.vMu

        third_doc_weights = doc_weights * 2*A_V_xi_squared / (vAlpha0s*(vAlpha0s+1.0))
        third_part = np.dot(self.vMu,np.dot(self.vAlpha * util.make_row_vector(third_doc_weights), self.vAlpha.T))

        document_sum = first_part - second_part - third_part
        return util.make_col_vector(A_V_xi * A_V_k0 * self.xi * self.vM) + self.k * document_sum

    def vMu_gradient_tan(self):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        A_V_k0 = util.bessel_approx(self.vocab_size, self.k0)
        gradient = self.vMu_gradient()
        for topic in range(self.num_topics):
            vMu_topic = self.vMu[:,topic]
            gradient[:,topic] = gradient[:,topic] - np.dot(vMu_topic, np.dot(vMu_topic.T, gradient[:,topic]))
        return gradient

    def do_update_vAlpha(self):
        util.optimize(self.vAlpha_likelihood, self.vAlpha_gradient, util.Parameter(self, 'vAlpha'), verbose = VERBOSE)

    def do_update_vMu(self, LAMBDA):
        def f():
            vMu_squared = np.sum(self.vMu ** 2, axis=0)
            return self.vMu_likelihood() - LAMBDA*np.sum((vMu_squared - 1.0) ** 2)

        def f_prime():
            vMu_squared = np.sum(self.vMu ** 2, axis=0)
            return self.vMu_gradient_tan() - LAMBDA*2.0*np.sum((vMu_squared - 1.0) * (2*self.vMu))

        util.optimize(f, f_prime, util.Parameter(self, 'vMu'), bounds=(-1.0,1.0), verbose = VERBOSE)
        self.vMu = util.l2_normalize(self.vMu)

    ####
    """ Alpha """
    ####
        
    def update_alpha(self):
        util.optimize(self.alpha_likelihood, self.alpha_likelihood_gradient, util.Parameter(self, 'alpha'), verbose = VERBOSE)
        
    def alpha_likelihood(self):
        alpha0 = np.sum(self.alpha)

        psi_vAlpha = psi(self.vAlpha)
        psi_vAlpha0s = psi(np.sum(self.vAlpha, axis=0))


        #likelihood = np.sum( ascolvector(self.alpha - 1) * psi_vAlpha ) \
        likelihood = np.sum(util.make_col_vector(self.alpha - 1) * psi_vAlpha ) \
                     - (alpha0 - self.num_topics)*np.sum(psi_vAlpha0s) \
                     + self.num_docs*gammaln(alpha0) \
                     - self.num_docs*np.sum(gammaln(self.alpha))
        return likelihood

    def alpha_likelihood_gradient(self):
        alpha0 = np.sum(self.alpha)
        valpha0s = np.sum(self.vAlpha, axis=0)

        return np.sum(psi(self.vAlpha), axis=1) - np.sum(psi(valpha0s)) \
            + self.num_docs*psi(alpha0) - self.num_docs*psi(self.alpha)
        
    ####
    """ Xi """
    ####
    
    def update_xi(self):
        util.optimize(self.xi_likelihood, self.xi_likelihood_gradient, util.Parameter(self, 'xi'), verbose = VERBOSE)
   
    def xi_likelihood(self):
        a_xi = util.bessel_approx(self.vocab_size, self.xi)
        a_k0 = util.bessel_approx(self.vocab_size, self.k0)
        #sum_of_rhos = sum(self.rho_batch())
        sum_rhos = sum(self.calc_rhos())
        
        return a_xi*self.xi * (a_k0*np.dot(self.vM.T, np.sum(self.vMu, axis=1)) - self.num_topics) \
            + self.k*sum_rhos

    def xi_likelihood_gradient(self):
        a_xi = util.bessel_approx(self.vocab_size, self.xi)
        a_prime_xi = util.bessel_approx_derivative(self.vocab_size, self.xi)
        a_k0 = util.bessel_approx(self.vocab_size, self.k0)

        # Delete?
        #sum_over_documents = sum(self.deriv_rho_xi())
        #return (a_prime_xi*self.xi + a_xi) * (a_k0*np.dot(self.vm.T, np.sum(self.vMu, axis=1)) - self.num_topics) \

        sum_over_documents = sum(self.rho_xi_grad())
        return (a_prime_xi*self.xi + a_xi) * (a_k0*np.dot(self.vM.T, np.sum(self.vMu, axis=1)) - self.num_topics) \
            + self.k*sum_over_documents

    """ Batch gradient of Rho_d's wrt xi. """            
    def rho_xi_grad(self):
        a_xi = util.bessel_approx(self.vocab_size, self.xi)
        deriv_a_xi = util.bessel_approx_derivative(self.vocab_size, self.xi)
        vAlpha0s = np.sum(self.vAlpha, axis=0)
        esns = self.expected_squared_norms()
        deriv_e_squared_norm_xis  = self.e_squared_norm_xi_grad()
        
        vMuTimesVAlphaDotDoc = np.sum(self.vAlpha * np.dot(self.vMu.T, self.documents), axis=0)

        deriv = deriv_a_xi * vMuTimesVAlphaDotDoc / (vAlpha0s * np.sqrt(esns)) \
            - a_xi/2 * vMuTimesVAlphaDotDoc / (vAlpha0s * esns**1.5) * deriv_e_squared_norm_xis
        return deriv

    """ Gradient of E[norms^2] wrt xi """
    def e_squared_norm_xi_grad(self):

        a_xi = util.bessel_approx(self.vocab_size, self.xi)
        deriv_a_xi = util.bessel_approx_derivative(self.vocab_size, self.xi)

        vAlpha0s = np.sum(self.vAlpha, axis=0)
        sum_vAlphas_squared = np.sum(self.vAlpha**2, axis=0)
        vMuVAlphaVMuVAlpha = np.sum(np.dot(self.vAlpha.T, np.dot(self.vMu.T, self.vMu)).T * self.vAlpha, axis=0)
        gradient = 2*a_xi*deriv_a_xi*(vMuVAlphaVMuVAlpha - sum_vAlphas_squared) / (vAlpha0s * (vAlpha0s + 1))
        return gradient

    def print_topics(self, top_words=15, bottom_words=15):
        for t in range(self.num_topics):
            util.log_message('TOPIC {}\nTop {} words:\n---------------\n'.format(t, top_words), self.log_file)

            sorted_word_indices = np.argsort(self.vMu[:, t])
            sorted_topic_weights = self.vMu[sorted_word_indices, t]
            sorted_topic_words = self.sorted_terms[sorted_word_indices]

            for i in range(top_words):
                util.log_message('{}: {:.4f}\n'
                                 .format(sorted_topic_words[self.vocab_size-(i+1)],
                                         sorted_topic_weights[self.vocab_size-(i+1)]),
                                 self.log_file)

            util.log_message('\nTOPIC {}\nBottom {} words:\n---------------\n'.format(t, bottom_words), self.log_file)
            for i in range(bottom_words):
                util.log_message('{}: {:.4f}\n'
                                 .format(sorted_topic_words[i],
                                         sorted_topic_weights[i]),
                                 self.log_file)
            util.log_message('\n', self.log_file)

        util.log_message('\n', self.log_file)

    def run(self):
        self.do_EM(ITERATIONS)
        # self.do_EM(1)

        import datetime
        date = datetime.date.today()
        corpus_name = os.path.split(self.corpus)[1][0:5]
        util.save_model(self, './data/models/{}_{}.pickle'.format(corpus_name, date))

    def get_topics(self):
        topics = []
        for t in range(self.num_topics):
            topic = []
            sorted_word_indices = np.argsort(self.vMu[:, t])
            sorted_topic_weights = self.vMu[sorted_word_indices, t]
            sorted_topic_words = self.sorted_terms[sorted_word_indices]
            for w in range(len(sorted_topic_words)):
                topic.append((sorted_topic_words[-w], sorted_topic_weights[-w]))

            topics.append(topic)

        return topics


    """ UPDATE METHODS"""


    def update_free_params(self):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)

        #topic_mean_sum = np.sum(self.vMu, axis =1 )
        #topic_mean_sum = np.sum(self.vMu)

        # sum_rhos = sum(self.calc_rhos())

        self.do_update_vAlpha()

        LAMBDA = 10.0 * self.vMu_likelihood()
        self.do_update_vMu(LAMBDA)

        self.vM = util.l2_normalize(self.k0 * self.m + A_V_xi * self.xi * np.sum(self.vMu, axis=1))

        """self.vM = util.l2_normalize(self.k0 * A_V_k0 * self.m +
                                    A_V_xi * A_V_k0 * self.xi *
                                    topic_mean_sum + 2 * LAMBDA * self.vM)
        """

        # Record vM update
        self.loss_updates["vM"].append(self.vM)

    def do_EM(self, max_iterations=100, print_topics_every=10):
        if not self.m_b is None:
            self.reset()
        self.print_topics(top_words=TOP, bottom_words=BOTTOM)
        for i in range(1, max_iterations + 1):
            util.log_message("\nITERATION {}\n".format(i), self.log_file)
            self.do_E()
            self.do_M()
            print(self.vMu.shape)
            if PRUNE_VOCAB and i > START_PRUNE_ITERATION:
                # Evenly distribute vocab to prune after 5 iterations
                limit_by = int(self.vocab_size_b*PERC_VOCAB_TO_KEEP/(max_iterations-START_PRUNE_ITERATION))
                self.prune_topics(limit_by=limit_by)
            if i % print_topics_every == 0:
                self.print_topics(top_words=TOP, bottom_words=BOTTOM)

        # OUTPUT
        # Calculate euclidean distances for m and vM
        #self.loss_updates["vM_raw"] = self.loss_updates["vM"][:]
        #self.loss_updates["m_raw"] = self.loss_updates["m"][:]
        self.calc_distance(self.loss_updates["vM"])
        self.calc_distance(self.loss_updates["m"])

        # Write out losses
        util.write_dictionary(self.loss_updates, self.loss_file)

        #if PRUNE_VOCAB:
        #    self.reset_vocab()

    def do_E(self):
        util.log_message("\tDoing expectation step of EM process...\n", self.log_file)
        self.update_free_params()


    def do_M(self):
        util.log_message("\tDoing maximization step of EM process...\n", self.log_file)
        self.update_model_params()
        return 0


    def calc_distance(self, l, distance = "euclidean"):
        #print([len(x) for x in l])
        for n in range(0, len(l)):
            if PRUNE_VOCAB and n > START_PRUNE_ITERATION:
                l[n] = np.delete(l[n], self.deleted_words[n-START_PRUNE_ITERATION-1])

            if n < len(l)-1:
                if distance == "euclidean":
                    l[n] = np.linalg.norm(l[n]-l[n+1])
                else:
                    l[n] = util.cosine_similarity(l[n], l[n+1])[0][0]

        # Delete the last item
        del l[-1]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', required=True,
                        help='Path to the location of your corpus. ' +
                             'This can be either a directory or a single file.')

    parser.add_argument('-C', '--corpus_codec', required=False,
                        help='utf-8, ansi, etc.')

    parser.add_argument('-t', '--topics', type=int, default=10,
                        help='Number of topics in model. Defaults to 10.')

    parser.add_argument('-s', '--stopwords',
                        default=None,
                        help='Optional path to a file containing stopwords.')

    parser.add_argument('-m', '--mode',
                        default='train', choices=['train', 'test'])

    parser.add_argument('-l', '--logfile',
                        default=None,
                        help='Optional path to save log information in. ' +
                             'If this is a directory, a file named log.txt will be created in it.')

    parser.add_argument('--loadfrom', help='Optional path to a pickle file containing a pre-trained model to load.')
    parser.add_argument('--saveto', help='Optional path to save the model to after training.')

    args = parser.parse_args()

    if args.loadfrom:
        model = util.load_model(args.loadfrom)
    else:
        model = SAM(args.corpus, args.topics, stopwords=args.stopwords, log_file=args.logfile, corpus_encoding=args.corpus_codec)

    if args.mode == 'train':
        # model.do_EM(max_iterations=1, print_topics_every=1)
        model.do_EM()

    # else:
    #     model.assign_topics()

    if args.saveto:
        util.save_model(model, args.saveto)
