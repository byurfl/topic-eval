from math import log,sqrt,pi
from scipy.optimize import fmin_tnc
from scipy.linalg import norm
import numpy as np
import pickle
import csv
import os

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
    # return l2_normalize_ours(data)
    return l2_normalize_his(data)

def l2_normalize_ours(data):
    arr_data = np.asarray(data)
    if len(arr_data.shape) == 1:
        l2_norm = np.fmax(np.sqrt(np.sum(arr_data * arr_data)), 10*EPS)
        return data / l2_norm

    elif len(arr_data.shape) == 2:
        col_norms = np.fmax(np.sqrt(np.sum(arr_data ** 2, axis=0)), 10*EPS)
        return data / make_row_vector(col_norms)
    else:
        raise ValueError('x should have one or two dimensions')

def column_norms(x):
    return np.sqrt(np.add.reduce((x * x), axis=0))


def l2_normalize_his(x):
    """
    Returns an L2-normalized version of the data in x.  If x is two-dimensional, each column of x is normalized.
    """
    x = np.asarray(x, dtype='float64')
    if x.ndim == 1:
        norm_ = np.fmax(norm(x), 100 * EPS)
        return x / norm_
    elif x.ndim == 2:
        norms = np.fmax(column_norms(x), 100 * EPS)
        return x / make_row_vector(norms)
    else:
        raise ValueError('x should have one or two dimensions')


def make_vector(matrix):
    return matrix.reshape(matrix.size)

def make_row_vector(matrix):
    return matrix.reshape(1, matrix.size)

def make_col_vector(matrix):
    return matrix.reshape((matrix.size, 1))

def ravel(matrix):
    return np.asarray(matrix).ravel()

def unravel(param, item):
    # param should be an instance of the Parameter class defined above
    # item should be a new value for the param
    if param.is_scalar:
        return np.asarray(item).item()
    else:
        return np.asarray(item).reshape(param.shape)

def optimize(function, func_deriv, param, bounds=[1e-4,None], disp=0, maxevals=100, verbose = True):
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

    # Record updates
    param.model.loss_updates[param.name].append(new_function_value - old_function_value)

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

    if verbose:
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


def write_dictionary(d, filename):
    from itertools import zip_longest
    import csv

    print(d)

    # replace space string values with empty lists
    for key, value in d.items():
        if value == ' ':
            d[key] = []

    with open(filename, "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(d.keys())
        writer.writerows(zip_longest(*d.values()))


def write_topics(topic_matrix, output_path, iteration_counter=None, pivot_table = False):
    output_file = output_path \
        if iteration_counter is None \
        else os.path.join(output_path, 'topics_' + str(iteration_counter) + '.csv')

    with open(output_file, 'w', newline='') as out:
        csv_out = csv.writer(out)
        if pivot_table:
            csv_out.writerow(['topic', 'name', 'num'])
            for t, topic in enumerate(topic_matrix):
                for word in topic:
                    csv_out.writerow(["TOPIC " + str(t)] + list(word))
        else:
            csv_out.writerow(['name', 'num'] * topic_matrix.shape[0])
            output_topics = topic_matrix[:]

            # Verify sort
            for topic in output_topics:
                topic.sort(key=lambda x: -x[1])

            output = zip(*output_topics)
            for word_row in output:
                csv_out.writerow([i for sub in word_row for i in sub])
