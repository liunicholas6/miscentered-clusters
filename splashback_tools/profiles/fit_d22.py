#*****************************************************************************
# Fitting routines for halo profiles in simulations
#*****************************************************************************
from random import uniform
import numpy as np
from numpy import log10
import scipy
from scipy.integrate import quad
import os
import pymultinest
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

def get_rsp(theta):
    """
    Computes location of minimum logarithmic derivative (i.e. splashback proxy)

    Parameters
    ----------
    theta: Nparam*1 array
        D22 model parameters NOT in the form
            [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
             log(rho_s), log(rho_0), s_e)]

    Returns
    -------
    float of radius corresponding to minimum log derivative
    """

    r_values = np.logspace(-1, 1, 10000)

    log_deriv = log_deriv_d22(theta, r_values)

    return r_values[np.argmin(log_deriv)]

#*****************************************************************************
# Fitting code
#*****************************************************************************

def fit_d22(data_vec, base_path, out_dir=None, n_live_points=500):
    """
    Function to fit d22 profile to a input data vector

    Parameters
    ----------
    theta: Nparam*1 array
        D22 model parameters NOT in the form
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
            DK22 model parameters in the form
                [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
                 log(rho_s), log(rho_0), s_e)]
        data_vec: 3 element tuple
            Tuple containing measured radii, densities, and covariance.

        Returns
        -------
            Float of the likelihood for given model parameter and data vector
        """
        # Unpack data and compute theory predicitons
        rho_thr = rho_d22(theta, r_data)

        # Compute likelihood
        diff = rho_data-rho_thr
        likelihood = -1/2 * np.dot(diff, np.linalg.solve(cov, diff))

        if (np.isnan(likelihood)==True):
            return -np.inf

        return likelihood


    #**************************************************************************
    # Sampling Functions
    #**************************************************************************

    # Format prior and likelihood for multinest
    def prior(cube, ndim=9, nparam=9):
        """
        Prior on parameters. Can be changed if chains aren't converging.
        Gaussian_prior and uniform_prior are functions defined in
        profile_calculations/utilities/multinest_priors that transform elements
        on the interval [0, 1] to gaussian or uniform distributions with the
        specified prameters.
        """
        cube[0] = uniform_prior(cube[0], log10(0.03), log10(0.4)) # lg_alpha
        cube[1] = uniform_prior(cube[1], log10(0.1), log10(10))  # lg_beta
        cube[2] = uniform_prior(cube[2], 1, 7)                 # lg_rho_s_over_rho_m
        cube[3] = uniform_prior(cube[3], log10(0.01), log10(0.45)) # lg_r_s
        cube[4] = uniform_prior(cube[4], log10(0.5), log10(3))     # lg_r_t
        cube[5] = uniform_prior(cube[5], 0, 1)                    # lg_d_1
        cube[6] = uniform_prior(cube[6], log10(0.01), log10(4))  # lg_s
        cube[7] = uniform_prior(cube[7], 1, log10(2000))           # lg_d_max
        cube[8] = uniform_prior(cube[8], -20, 20) #lg_rho_m

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

def sigma_d22(theta, r, l_max = 7):
    """
    Definition of halo profile model from Diemer 2022, projected into 2D with the miscentering profile from
    Shin et. al 2019

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

    l = np.linspace(-l_max, l_max)
    theta_0 = theta[0:9]
    f = theta[10]
    sigma_r = np.exp(theta[11])
    def sigma_0(r):
        return np.array([np.trapz(rho_d22(theta_0, np.sqrt(r_i**2 + l**2)), l) for r_i in r])
    def sigma_mis(r_mis):
        def integrand(phi, r, r_mis):
            return sigma_0(np.sqrt(r**2 + r_mis**2 + 2 * r * r_mis * np.cos(phi)))
    
        def sigma_mis_cond(r_mis, r):     
            phi = np.linspace(0, np.pi, 25)
            return np.trapz(integrand(phi, r, r_mis), phi) /(2 * np.pi)
        
        def prob_r_mis(r_mis):
            return r_mis/(sigma_r)**2 * np.exp(-(r_mis)**2 /(2*sigma_r**2))
        
        def integrand2(r_mis, r):
            return np.array([prob_r_mis(r_mis_i)*sigma_mis_cond(r_mis_i, r) for r_mis_i in r_mis])
        
        r_mis = np.linspace(0, 1)
        return np.array([np.trapz(integrand2(r_mis, r_i), r_mis) for r_i in r])
    
    return (1 - f) * sigma_0(r) + f * sigma_mis(r)

def fit_d22_2d(data_vec, base_path, out_dir=None, n_live_points=500):
    """
    Function to fit D22 profile to a input data vector

    Parameters
    ----------
    theta: Nparam*1 array
        d22 model parameters NOT in the form
            [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
             log(rho_s), log(rho_0), s_e, f, ln(sigma_r))]
    out_dir: str
        String specifying where to store chains from model fitting
    """

    #**************************************************************************
    # Likelihood and prior definitions
    # -> Formatted for emcee and pymultinest
    #**************************************************************************
    def ln_like2d(theta, r_data, sigma_data, cov):
        """
        Gaussian likelihood definition used for fitting D22 model

        Parameters
        ----------
        theta: Nparam*1 array
            DK14 model parameters in the form
                [log(alpha), log(beta), log(gamma), log(r_s), log(r_t),
                 log(rho_s), log(rho_0), s_e, f, ln(sigma_r))]
        data_vec: 3 element tuple
            Tuple containing measured radii, densities, and covariance.

        Returns
        -------
            Float of the likelihood for given model parameter and data vector
        """
        # Unpack data and compute theory predicitons
        sigma_thr = sigma_d22(theta, r_data)

        # Compute likelihood
        diff = sigma_data - sigma_thr
        likelihood = -1/2 * np.dot(diff, np.linalg.solve(cov, diff))

        if (np.isnan(likelihood)==True):
            return -np.inf

        return likelihood


    #**************************************************************************
    # Sampling Functions
    #**************************************************************************

    # Format prior and likelihood for multinest
    def prior(cube, ndim=11, nparam=11):
        """
        Prior on parameters. Can be changed if chains aren't converging.
        Gaussian_prior and uniform_prior are functions defined in
        profile_calculations/utilities/multinest_priors that transform elements
        on the interval [0, 1] to gaussian or uniform distributions with the
        specified prameters.
        """
        cube[0] = uniform_prior(cube[0], log10(0.03), log10(0.4)) # lg_alpha
        cube[1] = uniform_prior(cube[1], log10(0.1), log10(10))  # lg_beta
        cube[2] = uniform_prior(cube[2], 1, 7)                 # lg_rho_s_over_rho_m
        cube[3] = uniform_prior(cube[3], log10(0.01), log10(0.45)) # lg_r_s
        cube[4] = uniform_prior(cube[4], log10(0.5), log10(3))     # lg_r_t
        cube[5] = uniform_prior(cube[5], 0, 1)                    # lg_d_1
        cube[6] = uniform_prior(cube[6], log10(0.01), log10(4))  # lg_s
        cube[7] = uniform_prior(cube[7], 1, log10(2000))           # lg_d_max
        cube[8] = uniform_prior(cube[8], -20, 20) #lg_rho_m
        cube[9] = uniform_prior(cube[9], 0.2, 0.2)  #f
        cube[10] = gaussian_prior(cube[10], -1.19, 0.22**2)       #sigma_r

        return cube

    def loglike(cube, ndim=11, nparam=11):
        return ln_like2d(cube, *data_vec)

    print(TermColors.CGREEN+"Fitting profiles with multinest"+TermColors.CEND)
    pymultinest.run(loglike, prior, 11, n_live_points = n_live_points,
                    outputfiles_basename=base_path,
                    resume=False, verbose=False, evidence_tolerance = 0.01)
    # Save chains

    n_params  = 11
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