from math import log,sqrt,pi
from scipy.optimize import fmin_tnc
import numpy as np
import pickle


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
def bessel_approx_ours(v, z):
    ratio_z_v = z / v
    alpha = 1 + (ratio_z_v*ratio_z_v)
    eta = sqrt(alpha) + log(ratio_z_v) - log(1 + sqrt(alpha))

    return -log(sqrt(2*pi*v)) + (v*eta) - (0.25*log(alpha))
    #return -log(sqrt(2*pi*v)) + (v*(sqrt(1 + z / v*z / v) + log(z / v) - log(1 + sqrt(1 + z / v*z / v)))) - (0.25*log(1 + z / v*z / v))

def bessel_approx(v,z):
    #return bessel_approx_ours(v,z)
    return avk(v, z)

def avk(v,k):
    assert np.isscalar(v)
    assert np.isscalar(k)
    return (np.sqrt((v / k) ** 2 + 4) - v / k) / 2.0

def bessel_approx_derivative(v,z):
    #return bessel_approx_derivative(v,k)
    return avk_derivative(v, z)

def bessel_approx_derivative_ours(v,z):
    #-log(sqrt(2*pi*v)) + (v*eta) - (0.25*log(alpha))
    return (v * sqrt(z**2/v**2 + 1))/z - (0.5 * z)/(v**2 + z**2)
#https://www.wolframalpha.com/input/?i=derivative+of+-log(sqrt(2*pi*v))+%2B+(v*(sqrt(1+%2B+z+%2F+v*z+%2F+v)+%2B+log(z+%2F+v)+-+log(1+%2B+sqrt(1+%2B+z+%2F+v*z+%2F+v))))+-+(0.25*log(1+%2B+z+%2F+v*z+%2F+v))+wrt+z

def avk_derivative(v, k):
    #Derivative of the VMF mean resultant length wrt kappa.
    #a = AvK(v,k);
    #deriv = 1-a^2 - (v-1)/k*a;
    #return -1/2/(v^2/k^2+4)^(1/2)*v^2/k^3+1/2*v/k^2
    return -0.5 / (v**2/k**2+4)**0.5 * v**2/k**3 + 0.5*v/k**2

def l2_normalize(data):
    arr_data = np.asarray(data)
    if len(arr_data.shape) == 1:
        l2_norm = np.fmax(np.sqrt(np.sum(arr_data ** 2)), 10*EPS)
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

def unravel(param, item):
    # param should be an instance of the Parameter class defined above
    # item should be a new value for the param
    if param.is_scalar:
        return np.asarray(item).item()
    else:
        return np.asarray(item).reshape(param.shape)

def optimize(function, func_deriv, param, bounds=[1e-4,None], disp=0, maxevals=150):
    log_message("\t\tOptimizing parameter: {}\n".format(param.name), param.model.log_file)
    x0 = ravel(getattr(param.model, param.name))
    bounds = [bounds] * len(x0)

    def get_negative_func_and_func_deriv(param_list):
        # save away current value
        orig_value = getattr(param.model, param.name)
        # set new value
        setattr(param.model, param.name, unravel(param, param_list))
        func_value = -function()
        func_deriv_value = ravel(-func_deriv())
        setattr(param.model, param.name, orig_value)
        return func_value, func_deriv_value

    old_function_value = function()
    result,evals,rc = fmin_tnc(get_negative_func_and_func_deriv, x0=x0, bounds=bounds, disp=disp, maxfun=maxevals)
    setattr(param.model, param.name, unravel(param, result))
    new_function_value = function()

    if rc == -1:
        message = "Infeasible(lower bound > upper bound)"
    elif rc == 0:
        message = "Local minimum reached( | pg | ~ = 0)"
    elif rc == 1:
        message = "Converged (|f_n-f_(n-1)| ~= 0)"
    elif rc == 2:
        message = "Converged( | x_n - x_(n - 1) | ~ = 0)"
    elif rc ==  3:
        message = "Max. number of function evaluations reached"
    elif rc == 4:
        message = "Linear search failed"
    elif rc == 5:
        message = "All lower bounds are equal to the upper bounds"
    elif rc == 6:
        message = "Unable to progress"
    elif rc == 7:
        message = "User requested end of minimization"

    message = "\t\t(Message: " + message + ")\n\n"
    log_message("\t\tOptimized parameter: {} with improvement of {}\n".format(param.name, new_function_value - old_function_value), param.model.log_file)
    log_message("\t\tMinimizer returned code {} after {} iterations\n".format(rc, evals), param.model.log_file)
    log_message(message, param.model.log_file)

def log_message(message, file):
    print(message, end='')
    with open(file, mode='a', encoding='utf-8') as l:
        l.write(message)

def save_model(model, pickle_file):
    log_message('Saving model to ' + pickle_file + '\n', model.log_file)
    print('Current model state: ')
    print('vocab size: {}'.format(model.vocab_size))
    print('corpus size: {}'.format(model.num_docs))
    print('number of topics: {}'.format(model.num_topics))
    print('xi: {}'.format(model.xi))
    print('m: {}'.format(model.m))
    print('alpha: {}'.format(model.alpha))
    print('k0: {}'.format(model.k0))
    print('k: {}'.format(model.k))
    print('vAlpha: {}'.format(model.vAlpha))
    print('vMu: {}'.format(model.vMu))
    print('vM: {}'.format(model.vM))
    with open(pickle_file, mode='wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def load_model(pickle_file):
    print('Loading model from ' + pickle_file + '\n')
    with open(pickle_file, mode='rb') as file:
        return pickle.load(file)

def cosine_similarity(a, b):
    """
    Computes the cosine similarity of the columns of A with the columns of B.
    Returns a matrix X such that Xij is the cosine similarity of A_i with B_j.
    In the case that norm(A_i) = 0 or norm(B_j) = 0, this
    implementation will return X_ij = 0.  If norm(A_i) = 0 AND norm(B_i) = 0,
    then X_ii = 0 as well.
    """
    if a.ndim == 1:
        a = ascolvector(a)
    if b.ndim == 1:
        b = ascolvector(b)
    assert a.shape[0] == b.shape[0]

    return l2_normalize(a).T.dot(l2_normalize(b))

def ascolvector(x):
    return x.reshape(x.size, 1)
