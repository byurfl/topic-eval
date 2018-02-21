import algorithms.sam.math_util as math_util
import numpy as np

class SAM:
    def __init__(self):
        self.convergence = False
        self.epsilon = 0.001

    def analyze_data(self, corpus):
        self.vocabulary = []
        self.vocab_size = len(self.vocabulary)

    def update_model_params(self, xi, m, alpha, kappa_0, kappa):
        pass

    def update_free_params(self, v_alpha, v_mu, v_m):
        pass

    def do_EM(self):
        old_result = 0
        while not self.convergence:
            self.do_E()
            new_result = self.do_M()
            if new_result - old_result < self.epsilon:
                self.convergence = True

    def do_E(self):
        A_V_xi = math_util.bessel_approx(self.vocab_size, self.xi)
        A_V_k0 = math_util.bessel_approx(self.vocab_size, self.k0)

        topic_mean_sum = np.sum(self.vmu)
        delta_vm = self.k0*A_V_k0*self.m + A_V_xi*A_V_k0*self.xi*topic_mean_sum + 2*LAMBA*self.vm