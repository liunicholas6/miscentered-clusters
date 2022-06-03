#*****************************************************************************
# Collection of priors formatted for multinest
#*****************************************************************************

from math import sqrt
from scipy.special import erfcinv

def uniform_prior(c, x1, x2):
    return x1+c*(x2-x1)

def gaussian_prior(c, mu, sigma):
    if (c <= 1.0e-16 or (1.0-c) <= 1.0e-16):
        return -1.0e32
    else:
        return mu+sigma*sqrt(2.0)*erfcinv(2.0*(1.0-c))
