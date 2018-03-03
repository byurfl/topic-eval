import sys
sys.path.append(r"D:\PyCharm Projects\py-sam-master\topic-eval")

import src.algorithms.sam.util as util
from src.algorithms.sam.reader import Reader
import numpy as np
from scipy.special import gammaln, psi, polygamma

class SAM:
    def __init__(self, corpus, stopwords=None, topics=10):

        # when stopping criteria hasn't changed by more than
        # epsilon for a few iterations, stop topic discovery
        self.CONVERGENCE = False
        self.EPSILON = 0.001

        self.reader = Reader(stopwords)
        self.reader.read_corpus(corpus)

        self.vocabulary = self.reader.vocabulary
        self.vocab_size = len(self.vocabulary)
        self.documents = self.reader.documents
        self.num_docs = self.documents.shape[1]

        # initialize model hyperparameters
        self.num_topics = topics

        self.xi = 500.0
        self.m = util.l2_normalize(np.random.rand(self.vocab_size))
        self.alpha = np.random.rand(self.num_topics)
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

    def vMu_likelihood(self, A_V_xi, A_V_k0, sum_rhos):
        vM_dot_vMu = np.dot(self.vM.T, np.sum(self.vMu, axis=1))

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
        util.optimize(self.vAlpha_likelihood, self.vAlpha_gradient, util.Parameter(self, 'vAlpha'))

    def do_update_vMu(self, LAMBDA, A_V_xi, A_V_k0, sum_rhos):
        vMu_squared = np.sum(self.vMu ** 2, axis=0)
        def f():
            return self.vMu_likelihood(A_V_xi, A_V_k0, sum_rhos) - LAMBDA*np.sum((vMu_squared - 1.0) ** 2)

        def f_prime():
            return self.vMu_gradient_tan(A_V_xi, A_V_k0) - LAMBDA*np.sum((vMu_squared - 1.0) * (2*self.vMu))

        util.optimize(f, f_prime, util.Parameter(self, 'vMu'), bounds=(-1.0,1.0))
        self.vMu = util.l2_normalize(self.vMu)


    def update_free_params(self):
        A_V_xi = util.bessel_approx(self.vocab_size, self.xi)
        A_V_k0 = util.bessel_approx(self.vocab_size, self.k0)
        topic_mean_sum = np.sum(self.vMu)
        sum_rhos = sum(util.calc_rhos(A_V_xi, self.vMu, self.vAlpha, self.documents))

        self.do_update_vAlpha()

        LAMBDA = 15.0 * self.vMu_likelihood(A_V_xi, A_V_k0, sum_rhos)
        self.do_update_vMu(LAMBDA, A_V_xi, A_V_k0, sum_rhos)

        self.vM = util.l2_normalize(self.k0 * A_V_k0 * self.m +
                                    A_V_xi * A_V_k0 * self.xi *
                                    topic_mean_sum + 2 * LAMBDA * self.vM)

    def do_EM(self, max_iterations=100):
        for _ in range(max_iterations):
            print("\nITERATION {}".format(_))
            self.do_E()
            self.do_M()

    def do_E(self):
        print("\tDoing expectation step of EM process...")
        self.update_free_params()

    def do_M(self):
        print("\tDoing maximization step of EM process...")
        self.update_model_params()
        # TODO: return something meaningful
        return 0


    ####
    """ Alpha """
    ####
        
    def update_alpha(self):
        util.optimize(alpha_likelihood(), alpha_likelihood_gradient(), 'alpha')
        
    def alpha_likelihood(self):
        alpha0 = np.sum(self.alpha)

        psi_vAlpha = psi(self.vAlpha)
        psi_vAlpha0s = psi(np.sum(self.vAlpha, axis=0))

        likelihood = np.sum( ascolvector(self.alpha - 1) * psi_vAlpha ) \
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
        util.optimize(self.xi_likelihood(), self.xi_likelihood_gradient(), 'xi')
   
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
        sum_over_documents = sum(self.deriv_rho_xi())
        return (a_prime_xi*self.xi + a_xi) * (a_k0*np.dot(self.vm.T, np.sum(self.vMu, axis=1)) - self.num_topics) \
            + self.k*sum_over_documents

    """ Batch gradient of Rho_d's wrt xi. """            
    def rho_xi_grad(self):
        a_xi = util.bessel_approx(self.vocab_size, self.xi)
        deriv_a_xi = util.bessel_approx_derivative(self.vocab_size, self.xi)
        vAlpha0s = np.sum(self.vAlpha, axis=0)
        esns = self.e_squared_norm_batch()
        deriv_e_squared_norm_xis  = self.grad_e_squared_norm_xi()
        
        vMuTimesVAlphaDotDoc = np.sum(self.vAlpha * np.dot(self.vMu.T, self.v), axis=0)

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
