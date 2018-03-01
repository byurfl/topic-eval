from math import log,sqrt,pi
import numpy as np


# corresponds to avk() in original SAM code
# not sure why they do it differently, this is what's in
# the paper they reference
def bessel_approx(v, z):
    ratio_z_v = z / v
    alpha = 1 + (ratio_z_v*ratio_z_v)
    eta = sqrt(alpha) + log(ratio_z_v) - log(1 + sqrt(alpha))

    return -log(sqrt(2*pi*v)) + (v*eta) - (0.25*log(alpha))


def l2_normalize(data):
    arr_data = np.asmatrix(data)
    if len(arr_data.shape) == 1:
        l2_norm = np.sqrt(np.sum(np.multiply(arr_data * arr_data)))
        return data / l2_norm

    elif len(arr_data.shape) == 2:
        col_norms = np.sqrt(np.sum(np.multiply(arr_data, arr_data), axis=1))
        return data / col_norms

    else:
        raise Exception('Data may only have 1 or 2 dimensions')

def make_vector(matrix):
    return matrix.reshape(matrix.size)

def make_row_vector(matrix):
    return matrix.reshape(1, matrix.size)

def make_col_vector(matrix):
    return matrix.reshape((matrix.size, 1))

def expected_squared_norms(A_V_xi, vMu, vAlpha):
    vAlpha0s = np.sum(vAlpha, axis=0)
    vAlphas_squared = np.sum(vAlpha ** 2, axis=0)
    A_V_xi_squared = A_V_xi ** 2

    vMu_squared = np.dot(vMu.T, vMu)
    vMu_vAlpha_squared = np.sum(np.dot(vAlpha.T, vMu_squared).T * vAlpha, axis=0)

    result = (vAlpha0s + (1.0 - A_V_xi_squared) * vAlphas_squared + A_V_xi_squared * vMu_vAlpha_squared) / \
             (vAlpha0s * (vAlpha0s + 1.0))

    return result

def calc_rhos(A_V_xi, vMu, vAlpha, docs):
    expecteds = expected_squared_norms(A_V_xi, vMu, vAlpha)
    vAlpha0s = np.sum(vAlpha, axis=0)
    vMu_docs = vMu.T.dot(docs)

    return np.sum(vAlpha * make_row_vector(1.0 / vAlpha0s / np.sqrt(expecteds)) * vMu_docs, axis=0) * A_V_xi
