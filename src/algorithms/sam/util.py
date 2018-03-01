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


EPS = np.finfo('float64').eps

# corresponds to avk() in original SAM code
# not sure why they do it differently, this is what's in
# the paper they reference
def bessel_approx(v, z):
    ratio_z_v = z / v
    alpha = 1 + (ratio_z_v*ratio_z_v)
    eta = sqrt(alpha) + log(ratio_z_v) - log(1 + sqrt(alpha))

    return -log(sqrt(2*pi*v)) + (v*eta) - (0.25*log(alpha))
    #return -log(sqrt(2*pi*v)) + (v*(sqrt(1 + z / v*z / v) + log(z / v) - log(1 + sqrt(1 + z / v*z / v)))) - (0.25*log(1 + z / v*z / v))
    
def avk(v,z):
    return bessel_approx(v,z)
"""    
    assert np.isscalar(v)
    assert np.isscalar(k)
    return (np.sqrt((v / k) ** 2 + 4) - v / k) / 2.0
"""

def bessel_approx_derivative(v,k):
    -log(sqrt(2*pi*v)) + (v*eta) - (0.25*log(alpha))
    
def avk_derivative(v, k):
    #Derivative of the VMF mean resultant length wrt kappa.    
    #a = AvK(v,k);
    #deriv = 1-a^2 - (v-1)/k*a;
    #return -1/2/(v^2/k^2+4)^(1/2)*v^2/k^3+1/2*v/k^2
    return -0.5 / (v**2/k**2+4)**0.5 * v**2/k**3 + 0.5*v/k**2

    
def l2_normalize(data):
    arr_data = np.asarray(data)
    if len(arr_data.shape) == 1:
        l2_norm = np.fmax(np.sqrt(np.sum(arr_data * arr_data)), 10*EPS)
        return data / l2_norm

    elif len(arr_data.shape) == 2:
        col_norms = np.fmax(np.sqrt(np.sum(arr_data ** 2, axis=0)), 10*EPS)
        return data / make_row_vector(col_norms)

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
    vMu_vAlpha_squared_1 = np.dot(vAlpha.T, vMu_squared)
    vMu_vAlpha_squared_2 = vMu_vAlpha_squared_1.T * vAlpha
    vMu_vAlpha_squared = np.sum(vMu_vAlpha_squared_2, axis=0)

    result = (vAlpha0s + (1.0 - A_V_xi_squared) * vAlphas_squared + A_V_xi_squared * vMu_vAlpha_squared) / \
             (vAlpha0s * (vAlpha0s + 1.0))

    return result

def calc_rhos(A_V_xi, vMu, vAlpha, docs):
    expecteds = expected_squared_norms(A_V_xi, vMu, vAlpha)
    vAlpha0s = np.sum(vAlpha, axis=0)
    vMu_docs = vMu.T.dot(docs)

    result_1 = make_row_vector(1.0 / vAlpha0s / np.sqrt(expecteds))
    result_2 = vAlpha * result_1
    result_3 = result_2 * vMu_docs
    result_4 = np.sum(result_3, axis=0)
    result = result_4 * A_V_xi

    return result

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
