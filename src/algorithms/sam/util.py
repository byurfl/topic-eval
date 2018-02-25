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

