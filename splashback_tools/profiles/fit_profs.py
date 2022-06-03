#*****************************************************************************
# Fitting routines for halo profiles in simulations
#*****************************************************************************
import numpy as np
from numpy import log10
import scipy
import os
import pymultinest
from splashback_tools.utilities.multinest_priors import *
from splashback_tools.utilities.general import *
#*****************************************************************************
# Profile definitions
#*****************************************************************************
def rho_DK14(theta, r):
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


def log_deriv_DK14(theta, r):
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

    return r/rho_DK14(theta, r)*drho_dr


def get_rsp(theta):
    """
    Computes location of minimum logarithmic derivative (i.e. splashback proxy)

    Parameters
    ----------
    theta: Nparam*1 array
        DK14 model parameters in the form
            [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
             log(rho_s), log(rho_0), s_e)]

    Returns
    -------
    float of radius corresponding to minimum log derivative
    """

    r_values = np.logspace(-1, 1, 10000)

    log_deriv = log_deriv_DK14(theta, r_values)

    return r_values[np.argmin(log_deriv)]

#*****************************************************************************
# Fitting code
#*****************************************************************************

def fit_DK_14(data_vec, base_path, out_dir=None, n_live_points=500):
    """
    Function to fit DK14 profile to a input data vector

    Parameters
    ----------
    theta: Nparam*1 array
        DK14 model parameters in the form
            [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
             log(rho_s), log(rho_0), s_e)]
    out_dir: str
        String specifying where to store chains from model fitting
    """

    #**************************************************************************
    # Likelihood and prior definitions
    # -> Formatted for emcee and pymultinest
    #**************************************************************************
    def ln_like3d(theta,  r_data, rho_data, cov):
        """
        Gaussian likelihood definition used for fitting DK14 model

        Parameters
        ----------
        theta: Nparam*1 array
            DK14 model parameters in the form
                [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
                 log(rho_s), log(rho_0), s_e)]
        data_vec: 3 element tuple
            Tuple containing measured radii, densities, and covariance.

        Returns
        -------
            Float of the likelihood for given model parameter and data vector
        """
        # Unpack data and compute theory predicitons
        rho_thr = rho_DK14(theta, r_data)

        # Compute likelihood
        diff = rho_data-rho_thr
        likelihood = -1/2 * np.dot(diff, np.linalg.solve(cov, diff))

        if (np.isnan(likelihood)==True):
            return -np.inf

        return likelihood


    def ln_prior(theta):
        """
        Prior on DK14 model parameters. Currently based on arXiV:2111.06499

        Parameters
        ----------
        theta: Nparam*1 array
            DK14 model parameters in the form
                [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
                 log(rho_s), log(rho_0), s_e)]

        Returns
        -------
        Float
            Prior probability of input parameter theta.
        """

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


        # Top hat prior on r_s, r_t, s_e
        r_s_check = (r_s > 0.01) & (r_s < 5) # Change to 0.01
        r_t_check = (r_t > 0.01) & (r_t < 5) # Change to 0.01
        s_e_check = (s_e > 0.01) & (s_e < 10)

        # Assume wide priors for rho parameters
        rho_s_check = (lg_rho_0>-20) & (lg_rho_0<20)
        rho_0_check = (lg_rho_s>-20) & (lg_rho_s<20)
        rho_m_check = (lg_rho_m>-20) & (lg_rho_m<20)

        flat_prior_check = (r_s_check & r_t_check & s_e_check &
                            rho_s_check & rho_0_check & rho_m_check)

        if not flat_prior_check:
            return -np.inf
        else:
            return  (-0.5*(lg_alpha - log10(0.22))**2/0.6**2 -
                     0.5*(lg_beta - log10(4.0))**2/0.2**2 -
                     0.5*(lg_gamma - log10(6.0))**2/0.2**2)


    def ln_prob3d(theta, r_data, rho_data, cov):
        """
        Total probability definition (likelihood+prior)

        Parameters
        ----------
        theta: Nparam*1 array
            DK14 model parameters in the form
                [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
                 log(rho_s), log(rho_0), s_e)]
        data_vec: 3 element tuple
            Tuple containing measured radii, densities, and covariance.

        Returns
        -------
            Float of the total log probability for given model parameter and
            data vector
        """
        lp = ln_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp+ln_like3d(theta, r_data, rho_data, cov)


    #**************************************************************************
    # Sampling Functions
    #**************************************************************************

    # Set initial guess/boundaries and use minimizer to determine chain start
    init_theta = np.array([-0.50257944, 0.58536, 0.90564052,  -0.5, 0.35784823,
                            3.8549328, 8.051313892 , 0.1, 1.30773031])
    bounds = ((-2, 2),(-1.4, 3), (-1.2,3), (-1, 0.7), (-1, 0.7),
              (-20, 20), (-20, 20), (-20, 20), (0.01, 10))

    neg_ll3d = lambda *args: -ln_like3d(*args)
    res = scipy.optimize.minimize(neg_ll3d, init_theta, args=data_vec, method='SLSQP',
                                  options = {'maxiter':500}, bounds=bounds)




    # Format prior and likelihood for multinest
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
        cube[3] = uniform_prior(cube[3], -2, 1)             # lg_r_s
        cube[4] = uniform_prior(cube[4], -2, 1)             # lg_r_t
        cube[5] = uniform_prior(cube[5], -20, 20)           # lg_rho_s
        cube[6] = uniform_prior(cube[6], -20, 20)           # lg_rho_0
        cube[7] = uniform_prior(cube[7], -20, 20)           # lg_rho_m
        cube[8] = uniform_prior(cube[8], 0.01, 10)          # s_e

        return cube

    def loglike(cube, ndim=9, nparam=9):
        return ln_like3d(cube, *data_vec)

    print(TermColors.CGREEN+"Fitting profiles with multinest"+TermColors.CEND)
    pymultinest.run(loglike, prior, 9, n_live_points = n_live_points,
                    outputfiles_basename=base_path,
                    resume=False, verbose=False, evidence_tolerance = 0.01)
    # Save chains

    n_params  = 9
    multinest_analyzer = pymultinest.Analyzer(n_params, base_path)

    # Save files
    data = multinest_analyzer.get_data()[:,2:]
    weights = multinest_analyzer.get_data()[:,0]
    loglikes = multinest_analyzer.get_data().T[1]
    stacked_data = np.vstack((data.T, weights, loglikes)) # data; weights; loglikes

    if out_dir is not None:
        # Load analyzer
        print(TermColors.CGREEN+"Saving chains to: "+TermColors.CEND+out_dir)
        np.save(out_dir, stacked_data)
    else:
        return stacked_data
