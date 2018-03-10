# import sys
# sys.path.append(r"D:\PyCharm Projects\py-sam-master\topic-eval")

# Anchor words needs a home!
import os

if os.environ["COMPUTERNAME"] == 'DALAILAMA':
    os.environ["HOME"] = r"D:\PyCharm Projects\py-sam-master\topic-eval\data\corpus;"
    #print(os.getenv("HOME"))
    VERBOSE = False
    TOP =3
    BOTTOM =3
else:
    TOP = 15
    BOTTOM = 15
    VERBOSE = True

import src.algorithms.sam.util as util
from src.algorithms.sam.reader import Reader
import numpy as np
from scipy.special import gammaln, psi, polygamma



class SAM:
    def __init__(self, corpus, topics, stopwords=None, log_file=None):

        if log_file == None:
            self.log_file = corpus + '_log.txt'
        else:
            self.log_file = log_file

        if os.path.isdir(self.log_file):
            self.log_file = os.path.join(self.log_file, 'log.txt')

        # truncate log file if it exists, create it if it doesn't
        with open(self.log_file, mode='w', encoding='utf-8'):
            pass

        self.reader = Reader(stopwords)
        self.reader.read_corpus(corpus)

        self.vocabulary = self.reader.vocabulary
        self.vocab_size = len(self.vocabulary)

        self.sorted_terms = np.asarray(['' for _ in range(self.vocab_size)], dtype=object)
        for key, value in self.reader.terms_to_indices.items():
            self.sorted_terms[value] = key

        # with open(self.log_file, mode='w', encoding='utf-8') as log_out:
        #     for term in self.sorted_terms:
        #         log_out.write(term + '\n')

        self.documents = self.reader.documents
        self.num_docs = self.documents.shape[1]

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

    def update_model_params(self):
        self.m = util.l2_normalize(np.sum(self.vMu, axis=1))
        self.update_xi()
        self.update_alpha()


    def vAlpha_likelihood(self):
        alpha0 = np.sum(self.alpha)
        psi_vAlpha = psi(self.vAlpha)
        vAlpha0s = np.sum(self.vAlpha, axis=0)
        psi_vAlpha0s = psi(vAlpha0s)

        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        sum_rhos = sum(util.calc_rhos(A_V_xi, self.vMu, self.vAlpha, self.documents))

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

    def expected_square_norms_gradient(self, vAlpha0s, A_V_xi, square_norms):
        A_V_xi_squared = A_V_xi ** 2
        vMu_vAlpha_vMu = np.dot(self.vAlpha.T, np.dot(self.vMu.T, self.vMu))
        doc_weights = 1.0 / (vAlpha0s * (vAlpha0s + 1.0))
        gradient = 1 + 2 * (1-A_V_xi_squared) * self.vAlpha + 2 * A_V_xi_squared * vMu_vAlpha_vMu.T
        gradient = gradient - square_norms * util.make_row_vector(2 * vAlpha0s + 1.0)
        gradient = gradient * util.make_row_vector(doc_weights)

        return gradient

    def rho_vAlpha_gradient(self, vAlpha0s):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)

        expected_square_norms = util.expected_squared_norms(A_V_xi, self.vMu, self.vAlpha)
        square_norms_gradient = self.expected_square_norms_gradient(vAlpha0s, A_V_xi, expected_square_norms)

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

        rho_gradient = self.rho_vAlpha_gradient(vAlpha0s)

        gradient = util.make_col_vector(self.alpha - 1.0) * psi_vAlpha_gradient \
                    + self.k * rho_gradient - (self.vAlpha - 1.0) * psi_vAlpha_gradient

        row_constant = -psi_vAlpha0s_gradient * (alpha0 - self.num_topics) + \
                       psi_vAlpha0s_gradient * (vAlpha0s - self.num_topics)

        gradient = gradient + util.make_row_vector(row_constant)

        return gradient

    def vMu_likelihood(self, A_V_xi, A_V_k0):
        vM_dot_vMu = np.dot(self.vM.T, np.sum(self.vMu, axis=1))
        sum_rhos = sum(util.calc_rhos(A_V_xi, self.vMu, self.vAlpha, self.documents))

        return (A_V_xi * A_V_k0 * self.xi * vM_dot_vMu) + (self.k * sum_rhos)

    def vMu_gradient(self, A_V_xi, A_V_k0):
        A_V_xi_squared = A_V_xi ** 2
        squared_norms = util.expected_squared_norms(A_V_xi, self.vMu, self.vAlpha)
        vAlpha0s = np.sum(self.vAlpha, axis=0)

        first_part = np.dot(self.documents, (self.vAlpha * A_V_xi / util.make_row_vector(vAlpha0s * np.sqrt(squared_norms))).T)
        doc_weights = A_V_xi / vAlpha0s / (2 * squared_norms ** (3.0/2.0)) * (self.vAlpha * np.dot(self.vMu.T, self.documents)).sum(axis=0).T

        second_doc_weights = 2*(1-A_V_xi_squared) / (vAlpha0s*(vAlpha0s+1.0))
        second_part = np.sum(doc_weights * second_doc_weights) * self.vMu

        third_doc_weights = doc_weights * 2*A_V_xi_squared / (vAlpha0s*(vAlpha0s+1.0))
        third_part = np.dot(self.vMu,np.dot(self.vAlpha * util.make_row_vector(third_doc_weights), self.vAlpha.T))

        document_sum = first_part - second_part - third_part
        return util.make_col_vector(A_V_xi * A_V_k0 * self.xi * self.vM) + self.k * document_sum

    def vMu_gradient_tan(self, A_V_xi, A_V_k0):
        gradient = self.vMu_gradient(A_V_xi, A_V_k0)
        for topic in range(self.num_topics):
            vMu_topic = self.vMu[:,topic]
            gradient[:,topic] = gradient[:,topic] - np.dot(vMu_topic, np.dot(vMu_topic.T, gradient[:,topic]))
        return gradient

    def do_update_vAlpha(self):
        util.optimize(self.vAlpha_likelihood, self.vAlpha_gradient, util.Parameter(self, 'vAlpha'), verbose = VERBOSE)

    def do_update_vMu(self, LAMBDA, A_V_xi, A_V_k0):
        def f():
            vMu_squared = np.sum(self.vMu ** 2, axis=0)
            return self.vMu_likelihood(A_V_xi, A_V_k0) - LAMBDA*np.sum((vMu_squared - 1.0) ** 2)

        def f_prime():
            vMu_squared = np.sum(self.vMu ** 2, axis=0)
            return self.vMu_gradient_tan(A_V_xi, A_V_k0) - LAMBDA*2.0*np.sum((vMu_squared - 1.0) * (2*self.vMu))

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
        sum_rhos = sum(util.calc_rhos(a_xi, self.vMu, self.vAlpha, self.documents))
        
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
        esns = util.expected_squared_norms(a_xi, self.vMu, self.vAlpha)
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
        self.do_EM(5)
        # self.do_EM(1)
        import datetime
        date = datetime.datetime.now()
        year = date.year
        month = date.month
        day = date.day

        util.save_model(self, './data/models/enron_{}{:>02}{:>02}.pickle'.format(year, month, day))

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
        A_V_k0 = util.bessel_approx(self.vocab_size, self.k0)
        topic_mean_sum = np.sum(self.vMu)
        # sum_rhos = sum(util.calc_rhos(A_V_xi, self.vMu, self.vAlpha, self.documents))

        self.do_update_vAlpha()

        LAMBDA = 15.0 * self.vMu_likelihood(A_V_xi, A_V_k0)
        self.do_update_vMu(LAMBDA, A_V_xi, A_V_k0)

        self.vM = util.l2_normalize(self.k0 * A_V_k0 * self.m +
                                    A_V_xi * A_V_k0 * self.xi *
                                    topic_mean_sum + 2 * LAMBDA * self.vM)


    def do_EM(self, max_iterations=100, print_topics_every=10):
        self.print_topics(top_words=TOP, bottom_words=BOTTOM )
        for i in range(1, max_iterations + 1):
            util.log_message("\nITERATION {}\n".format(i), self.log_file)
            self.do_E()
            self.do_M()

            if i % print_topics_every == 0:
                self.print_topics(top_words=TOP, bottom_words=BOTTOM )


    def do_E(self):
        util.log_message("\tDoing expectation step of EM process...\n", self.log_file)
        self.update_free_params()


    def do_M(self):
        util.log_message("\tDoing maximization step of EM process...\n", self.log_file)
        self.update_model_params()
        # TODO: return something meaningful
        return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', required=True,
                        help='Path to the location of your corpus. ' +
                             'This can be either a directory or a single file.')

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
        model = SAM(args.corpus, args.topics, stopwords=args.stopwords, log_file=args.logfile)

    if args.mode == 'train':
        # model.do_EM(max_iterations=1, print_topics_every=1)
        model.do_EM()

    # else:
    #     model.assign_topics()

    if args.saveto:
        util.save_model(model, args.saveto)
