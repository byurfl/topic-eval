from math import log,sqrt,pi
from scipy.optimize import fmin_tnc
import numpy as np


class Parameter:
    def __init__(self, model, item_name):
        item = getattr(model, item_name)
        self.is_scalar = np.isscalar(item)
        self.shape = item.shape if not self.is_scalar else None
        self.model = model
        self.name = item_name


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
    vAlphas_squared = np.sum(np.square(vAlpha), axis=0)
    A_V_xi_squared = np.square(A_V_xi)

    vMu_squared = np.dot(vMu.T, vMu)
    vMu_vAlpha_squared = np.sum(np.multiply(np.dot(vAlpha.T, vMu_squared).T, vAlpha), axis=0)

    result = (vAlpha0s + np.multiply((1.0 - A_V_xi_squared), vAlphas_squared) + np.multiply(A_V_xi_squared, vMu_vAlpha_squared)) / \
             np.multiply(vAlpha0s, vAlpha0s + 1.0)

    return result

def calc_rhos(A_V_xi, vMu, vAlpha, docs):
    expecteds = expected_squared_norms(A_V_xi, vMu, vAlpha)
    vAlpha0s = np.sum(vAlpha, axis=0)
    vMu_docs = vMu.T.dot(docs)

    return np.multiply(np.sum(np.multiply(np.multiply(vAlpha, make_row_vector(1.0 / vAlpha0s / np.sqrt(expecteds))), vMu_docs), axis=0), A_V_xi)

def ravel(matrix):
    return np.asarray(matrix).ravel()

def unravel(item):
    # item should be an instance of the Parameter class defined above
    if item.is_scalar:
        return np.asarray(item).item()
    else:
        return np.asarray(item).reshape(item.shape)

def optimize(function, func_deriv, param, bounds, disp=0, maxevals=150):
    x0 = ravel(getattr(param.model, param.name))
    bounds = [bounds] * len(x0)

    def get_negative_func_and_func_deriv(param_list):
        # save away current value
        orig_value = getattr(param.model, param.name)
        # set new value
        setattr(param.model, param.name, unravel(param_list))
        func_value = -function()
        func_deriv_value = ravel(-func_deriv())
        setattr(param.model, param.name, orig_value)
        return func_value, func_deriv_value

    old_function_value = function()
    result,_,_ = fmin_tnc(function, x0=x0, bounds=bounds, disp=disp, maxfun=maxevals)
    setattr(param.model, param.name, unravel(result))
    new_function_value = function()
    print("Optimized parameter: {} with improvement {}".format(param.name, new_function_value - old_function_value))
