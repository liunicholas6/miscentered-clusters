#*****************************************************************************
# Fitting routines for halo profiles in simulations
#*****************************************************************************
import numpy as np
import pymultinest
from numpy import log10
from scipy.integrate import quad
from splashback_tools.utilities.multinest_priors import *
from splashback_tools.utilities.general import *

#*****************************************************************************
# Profile definitions
#*****************************************************************************
def rho_d22(theta, r):
    """
    Definition of halo profile model from Diemer 2022

    Parameters
    ----------
    theta: Nparam*1 array
        D22 model parameters in the form
            [log(alpha), log(beta), log(rho_s/rho_m), log(r_s), log(r_t), log(d_1), log(s), log(d_max)]
    r: N*1 array
        Radial values (in Mpc/h) at which to compute the profile

    Returns
    -------
    N*1 array of density profile values
    """

    # Unpack element-by-element so multinest doesn't complain
    lg_alpha = theta[0]
    lg_beta = theta[1]
    lg_rho_s_over_rho_m = theta[2]
    lg_r_s  = theta[3]
    lg_r_t = theta[4]
    lg_d_1 = theta[5]
    lg_s   = theta[6]
    lg_d_max = theta[7]
    lg_rho_m = theta[8]

    alpha = 10.**lg_alpha
    beta = 10.**lg_beta
    r_s = 10.**lg_r_s
    r_t = 10.**lg_r_t
    rho_s_over_rho_m = 10**lg_rho_s_over_rho_m
    d_1 = 10**lg_d_1
    d_max = 10**lg_d_max
    s = 10**lg_s    
    rho_m = 10**lg_rho_m


    def rho_orbit(r):
        exp_arg = -(2/alpha)*((r/r_s)**alpha - 1) - (1/beta)*((r/r_t)**beta - (r_s/r_t)**beta )
        return rho_s_over_rho_m*np.exp(exp_arg)

    def rho_infall(r):
        return (d_1/((d_1/d_max)**2 + r**(2*s))**0.5 + 1)

    return rho_m * (rho_orbit(r) + rho_infall(r))

def rho_dk14(theta, r):
    """
    Definition of halo profile model from Diemer Kravstov 2014

    Parameters
    ----------
    theta: Nparam*1 array
        DK14 model parameters in the form
            [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
             log(rho_s), log(rho_0), s_e)]
    r: N*1 array
        Radial values (in Mpc/h) at which to compute the profile

    Returns
    -------
    N*1 array of density profile values
    """

    r_0 = 5 # Fix r_0 to 1.5 Mpc/h


    # Unpack element-by-element so multinest doesn't complain
    lg_alpha = theta[0]
    lg_beta  = theta[1]
    lg_gamma = theta[2]
    lg_r_s   = theta[3]
    lg_r_t   = theta[4]
    lg_rho_s = theta[5]
    lg_rho_0 = theta[6]
    lg_rho_m = theta[7]
    s_e      = theta[8]

    alpha = 10.**lg_alpha
    beta = 10.**lg_beta
    gamma = 10.**lg_gamma
    r_s = 10.**lg_r_s
    r_t = 10.**lg_r_t
    rho_s = 10**lg_rho_s
    rho_0 = 10**lg_rho_0
    rho_m = 10**lg_rho_m


    def rho_inner(r):
        exp_arg = -2/alpha*((r/r_s)**alpha-1)
        return rho_s*np.exp(exp_arg)

    def f_trans(r):
        return (1+(r/r_t)**beta)**(-gamma/beta)

    def rho_outer(r):
        return rho_m*(rho_0*(r/r_0)**(-s_e)+1)

    return rho_inner(r)*f_trans(r)+rho_outer(r)

def sigma(rho):
    """
    Projects a 3d density function into 2d

    Parameters:
    -----------
    rho: lambda (theta: Nparam*1 array, r: N*1 array) -> N*1 array

    Returns:
    --------
    output: lambda(theta: Nparam*1 array, r: N*1 array, l_max: float) -> N* array
    """
    def output(theta, r, l_max = 7):
        def f(r):
            res, _ = quad(lambda l: rho(theta, np.sqrt(r**2 + l**2)), -l_max, l_max)
            return res
        return f(r) if np.isscalar(r) else np.array([f(r_i) for r_i in r])
    return output

#*****************************************************************************
# Log derivs
#*****************************************************************************   

def log_deriv_d22(theta, r):
    """
    Logarithmic derivative of profile Diemer 2022. Uses the relation
    dlogy/dlogx = x/y*dy/dx

     Parameters
    ----------
    theta: Nparam*1 array
        D22 model parameters in the form
            [log(alpha), log(beta), log(rho_s/rho_m), log(r_s), log(r_t), log(d_1), log(s), log(d_max)]
    r: N*1 array
        Radial values (in Mpc/h) at which to compute the profile

    Returns
    -------
    N*1 array of density profile values
    """

    # Unpack element-by-element so multinest doesn't complain
    lg_alpha = theta[0]
    lg_beta = theta[1]
    lg_rho_s_over_rho_m = theta[2]
    lg_r_s  = theta[3]
    lg_r_t = theta[4]
    lg_d_1 = theta[5]
    lg_s   = theta[6]
    lg_d_max = theta[7]
    lg_rho_m = theta[8]

    alpha = 10.**lg_alpha
    beta = 10.**lg_beta
    r_s = 10.**lg_r_s
    r_t = 10.**lg_r_t
    rho_s_over_rho_m = 10**lg_rho_s_over_rho_m
    d_1 = 10**lg_d_1
    d_max = 10**lg_d_max
    s = 10**lg_s
    rho_m = 10**lg_rho_m

    def drho_infall_dr(r):
        return - d_1 * s * r ** (2 * s -1) / ((d_1/d_max) ** 2 + r ** (2 * s)) ** 1.5
    
    def rho_orbit(r):
        exp_arg = -(2/alpha)*((r/r_s)**alpha - 1) - (1/beta)*((r/r_t)**beta - (r_s/r_t)**beta )
        return rho_s_over_rho_m*np.exp(exp_arg)

    def drho_orbit_dr(r):
        sprime = -2 * r ** (alpha - 1) / r_s ** alpha - r ** (beta - 1) / r_t ** beta
        return sprime * rho_orbit(r)

    drho_dr = rho_m * (drho_infall_dr(r) + drho_orbit_dr(r))

    return drho_dr * r / rho_d22(theta, r)

def log_deriv_dk14(theta, r):
    """
    Logarithmic derivative of profile Diemer Kravstov 2014. Uses the relation
    dlogy/dlogx = x/y*dy/dx

    Parameters
    ----------
    theta: Nparam*1 array
        DK14 model parameters in the form
            [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
             log(rho_s), log(rho_0), s_e)]
    r: N*1 array
        Radial values (in Mpc/h) at which to compute the profile

    Returns
    -------
    N*1 array of density profile values
    """

    r_0 = 5 # Fix r_0 to 1.5 Mpc/h

    # Unpack element-by-element so multinest doesn't complain
    lg_alpha = theta[0]
    lg_beta  = theta[1]
    lg_gamma = theta[2]
    lg_r_s   = theta[3]
    lg_r_t   = theta[4]
    lg_rho_s = theta[5]
    lg_rho_0 = theta[6]
    lg_rho_m = theta[7]
    s_e      = theta[8]

    alpha = 10.**lg_alpha
    beta = 10.**lg_beta
    gamma = 10.**lg_gamma
    r_s = 10.**lg_r_s
    r_t = 10.**lg_r_t
    rho_s = 10**lg_rho_s
    rho_0 = 10**lg_rho_0
    rho_m = 10**lg_rho_m


    def rho_inner(r):
        exp_arg = -2/alpha*((r/r_s)**alpha-1)
        return rho_s*np.exp(exp_arg)

    def f_trans(r):
        return (1+(r/r_t)**beta)**(-gamma/beta)

    def drho_inner_dr(r):
        exp_arg = -2/alpha*((r/r_s)**alpha-1)
        return rho_s*np.exp(exp_arg)*(-2/r_s*(r/r_s)**(alpha-1))

    def df_trans_dr(r):
        return -gamma/beta*(1+(r/r_t)**beta)**(-gamma/beta-1)*beta/r_t*(r/r_t)**(beta-1)

    def drho_outer_dr(r):
        return -rho_m*s_e*rho_0*(r/r_0)**(-s_e-1)/r_0


    drho_dr = rho_inner(r)*df_trans_dr(r)+drho_inner_dr(r)*f_trans(r)+drho_outer_dr(r)

    return r/rho_dk14(theta, r)*drho_dr

#*****************************************************************************
# Fitting code
#*****************************************************************************

def fit_profile(data_vec, base_path, prof_type = "d22", is_2d = False, avg_R200m = None, n_live_points=500):
    """
    Returns a fit density profile
    
    Parameters:
    -----------
    data_vec: 3 element tuple
        Tuple containing measured radii, densities, and covariance.
    base_path: str
        String containing path to run multinest in
    prof: str
        Profile to fit. Currently supported are "d22" and "dk14"
    is_2d: bool
        True if surface density, False (default) if volume density
    avg_R200m: float
        Average R200m (in any unit) to set prior bounds. None if units of R are
        R/R200m (default)
    n_live_points: float
        Number of live points (for multinest)
    """

    implemented_profs = ['d22', 'dk14']

    assert prof_type in implemented_profs, \
        (TermColors.CRED+"Profile type specifier must be\n->"+'\n->'.join(str(p) for p in implemented_profs)+
         TermColors.CEND)


    lg_avg_R200m = 0 if avg_R200m is None else log10(avg_R200m)

    # Picking the correct profile density function and prior
    if prof_type == "d22":
        dens_f = rho_d22
        n_params = 9
        def prior(cube, ndim=9, nparam=9):
            cube[0] = uniform_prior(cube[0], log10(0.03), log10(0.4)) # lg_alpha
            cube[1] = uniform_prior(cube[1], log10(0.1), log10(10))  # lg_beta
            cube[2] = uniform_prior(cube[2], 1, 7)                 # lg_rho_s_over_rho_m
            cube[3] = uniform_prior(cube[3], log10(0.01) + lg_avg_R200m, log10(0.45) + lg_avg_R200m) # lg_r_s
            cube[4] = uniform_prior(cube[4], log10(0.5) + lg_avg_R200m, log10(3) + lg_avg_R200m)     # lg_r_t
            cube[5] = uniform_prior(cube[5], 0, 1)                    # lg_d_1
            cube[6] = uniform_prior(cube[6], log10(0.01), log10(4))  # lg_s
            cube[7] = uniform_prior(cube[7], 1, log10(2000))           # lg_d_max
            cube[8] = uniform_prior(cube[8], -20, 20) #lg_rho_m

    elif prof_type == "dk14":
        dens_f = rho_dk14
        n_params = 9
        def prior(cube, ndim=9, nparam=9):
            """
            Prior on parameters. Can be changed if chains aren't converging.
            Gaussian_prior and uniform_prior are functions defined in
            profile_calculations/utilities/multinest_priors that transform elements
            on the interval [0, 1] to gaussian or uniform distributions with the
            specified prameters.
            """
            cube[0] = gaussian_prior(cube[0], log10(0.22), 0.6) # lg_alpha
            cube[1] = gaussian_prior(cube[1], log10(4.0), 0.2)  # lg_beta
            cube[2] = gaussian_prior(cube[2], log10(6.0), 0.2)  # lg_gamma
            cube[3] = uniform_prior(cube[3], -2 + lg_avg_R200m, 1 + lg_avg_R200m)             # lg_r_s
            cube[4] = uniform_prior(cube[4], -2 + lg_avg_R200m, 1 + lg_avg_R200m)             # lg_r_t
            cube[5] = uniform_prior(cube[5], -20, 20)           # lg_rho_s
            cube[6] = uniform_prior(cube[6], -20, 20)           # lg_rho_0
            cube[7] = uniform_prior(cube[7], -20, 20)           # lg_rho_m
            cube[8] = uniform_prior(cube[8], 0.01, 10)          # s_e
            return cube
    
    if is_2d:
        dens_f = sigma(dens_f)

    def ln_likelihood(theta, r_data, dens_data, cov):
        dens = dens_f(theta, r_data)
        diff = dens - dens_data
        likelihood = -1/2 * np.dot(diff, np.linalg.solve(cov, diff))
        return -np.inf if np.isnan(likelihood) else likelihood

    def loglike(cube, ndim = n_params, nparam = n_params):
        return ln_likelihood(cube, *data_vec)

    print(TermColors.CGREEN+"Fitting profiles with multinest"+TermColors.CEND)
    pymultinest.run(loglike, prior, n_params, n_live_points = n_live_points,
                    outputfiles_basename=base_path,
                    resume=False, evidence_tolerance = 0.01)

    multinest_analyzer = pymultinest.Analyzer(n_params, base_path)

    data = multinest_analyzer.get_data()
    thetas = data[:,2:]
    weights = data[:,0]
    loglikes = data.T[1]
    return np.vstack((thetas.T, weights, loglikes)) # data; weights; loglikes