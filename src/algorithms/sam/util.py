from math import log,sqrt,pi

# corresponds to avk() in original SAM code
# not sure why they do it differently, this is what's in
# the paper they reference
def bessel_approx(v, z):
    ratio_z_v = z / v
    alpha = 1 + (ratio_z_v*ratio_z_v)
    eta = sqrt(alpha) + log(ratio_z_v) - log(1 + sqrt(alpha))

    return -log(sqrt(2*pi*v)) + (v*eta) - (0.25*log(alpha))

