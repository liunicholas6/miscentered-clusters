#*****************************************************************************
# Calculations routines for halo profiles in simulations
#*****************************************************************************

import numpy as np
import scipy
import sys
import os
from scipy.stats import binned_statistic
from tqdm import tqdm
import joblib


# Set number of threads for parallelization
if 'OMP_NUM_THREADS' in os.environ.keys():
    n_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    n_threads = os.cpu_count()


#*****************************************************************************
# Halo analysis routines
#*****************************************************************************

def get_particles(halo_center, dm_part_pos, R, boxsize):
    """
    Function to return all particles within R of a halo_center while accounting
    for periodic boundary conditions. All inputs must be in the same units.

    Parameters
    ----------
    halo_center: 3*1 numpy array c
        Halo center position
    dm_part_pos: Ndm*3 numpy array
        Positions of all relevant dm particles
    R          : float
        Radius to find particles around center
    boxsize    : float
        Boxsize for periodic boundary calculations

    Returns
    -------
    (n, 3) numpy array
        Positions of all particles within R of halo_center
    """

    # Apply cubic filter to limit searches
    delta_x = np.abs(dm_part_pos.T[0]-halo_center[0])
    delta_y = np.abs(dm_part_pos.T[1]-halo_center[1])
    delta_z = np.abs(dm_part_pos.T[2]-halo_center[2])

    box_filt = np.where(((delta_x <= R) | (boxsize-delta_x <= R)) &
                        ((delta_y <= R) | (boxsize-delta_y <= R)) &
                        ((delta_z <= R) | (boxsize-delta_z <= R)))

    dm_coords_filt = dm_part_pos[box_filt]

    # Compute radius from halo center and account for periodic BC's
    dev = dm_coords_filt-halo_center

    for ind, q in enumerate(dev.T):
        q = np.where(np.abs(q) > 0.5 * boxsize, boxsize-np.abs(q), q)
        dev.T[ind] = q

    # Return particles with r<R
    r = np.linalg.norm(dev, axis=1)

    return dm_coords_filt[np.where(r < R)]


def compute_weighted_profile(halo_center, dm_part_pos, bins, boxsize,
                             weights, R200m=1, bin_stat="mean"):
    """
    Compute a single 3D profile of a specified input quantity. Note this
    function is not used for computing density profiles, but instead for
    computing quantities such as the velocity dispersion profile where the
    volume of shells is irrelevant

    Parameters
    ----------
    halo_center: 3*1 float array
        Halo center position
    dm_part_pos: Ndm*3 float array
        Positions of all relevant dm particles
    bins: Nbins*1 float array
        Array containing edges of radial bins for profile calculations
    boxsize    : float
        Boxsize for periodic BC calculations
    weights    : Ndm*1 float Array
        Array containing weights for each particle. This is usually the quantity
        for which the profile is computed
    R200m      : float
        Optional float containing R200m for rescaling in bins of r/R200m
    bin_stat   : str
        Optional string specifying whether to compute mean or median profile

    Returns
    -------
    avg_r: Nbin*1 array
        Average radial position of each density bin
    avg_prof: Nbin*1 array
        Aveage profile value in each radial bin
    """


    # Speed up calculation by excluding box around max radius
    delta_x = np.abs(dm_part_pos.T[0]-halo_center[0])
    delta_y = np.abs(dm_part_pos.T[1]-halo_center[1])
    delta_z = np.abs(dm_part_pos.T[2]-halo_center[2])

    R_max = R200m*np.max(bins)

    box_filt = np.where(((delta_x <= R_max) | (boxsize-delta_x <= R_max)) &
                        ((delta_y <= R_max) | (boxsize-delta_y <= R_max)) &
                        ((delta_z <= R_max) | (boxsize-delta_z <= R_max)))

    dm_part_pos = dm_part_pos[box_filt]
    weights = weights[box_filt]

    # Account for periodic bc
    dev = dm_part_pos-halo_center

    for ind, q in enumerate(dev.T):
        q = np.where(np.abs(q) > 0.5 * boxsize, boxsize-np.abs(q), q)
        dev.T[ind] = q

    # Compute and return profile
    r = np.linalg.norm(dev, axis=1)/R200m

    avg_prof, avg_r, _ = binned_statistic(r, weights, bin_stat, bins)
    avg_r = 1/2*(avg_r[1:]+avg_r[0:-1])

    return avg_r, avg_prof


def compute_density_profile(halo_center, dm_part_pos, bins, boxsize,
                            weights=None, R200m=1):
    """
    Compute a single 3D density profile given a set of input positions

    Parameters
    ----------
    halo_center: 3*1 float array
        Halo center position
    dm_part_pos: Ndm*3 float array
        Positions of all relevant dm particles
    bins: Nbins*1 float array
        Array containing edges of radial bins for profile calculations
    boxsize    : float
        Boxsize for periodic BC calculations
    weights    : Ndm*1 float Array
        Optional array containing weights for each particle (e.g. for volume
        or mass weighted quantities)
    R200m      : float
        Optional float containing R200m for rescaling in bins of r/R200m

    Returns
    -------
    avg_r: Nbin*1 array
        Average radial position of each density bin
    avg_rho: Nbin*1 array
        Average density in each radial bin
    """

    if weights is None:
        #print("-> no weights specified -- assuming number density profile.")
        weights = np.ones(len(dm_part_pos))
    elif isinstance(weights, (int, float)):
        #print("-> single number specified for profile weighting.")
        weights = weights*np.ones(len(dm_part_pos))


    # Speed up calculation by excluding box around max radius
    delta_x = np.abs(dm_part_pos.T[0]-halo_center[0])
    delta_y = np.abs(dm_part_pos.T[1]-halo_center[1])
    delta_z = np.abs(dm_part_pos.T[2]-halo_center[2])

    R_max = R200m*np.max(bins)

    box_filt = np.where(((delta_x <= R_max) | (boxsize-delta_x <= R_max)) &
                        ((delta_y <= R_max) | (boxsize-delta_y <= R_max)) &
                        ((delta_z <= R_max) | (boxsize-delta_z <= R_max)))

    dm_part_pos = dm_part_pos[box_filt]
    weights = weights[box_filt]

    # Account for periodic bc
    dev = dm_part_pos-halo_center

    for ind, q in enumerate(dev.T):
        q = np.where(np.abs(q) > 0.5 * boxsize, boxsize-np.abs(q), q)
        dev.T[ind] = q

    # Compute and return profile
    r = np.linalg.norm(dev, axis=1)/R200m
    counts, r_edges = np.histogram(r, bins=bins, weights=weights)

    avg_r = 1/2*(r_edges[1:]+r_edges[0:-1])
    dV = 4/3*np.pi*(r_edges[1:]**3-r_edges[0:-1]**3)

    avg_rho = counts/dV*R200m**3

    return avg_r, avg_rho


def compute_veloc_disp_prof(halo_center, dm_part_pos, dm_part_vel, bins,
                            boxsize, R200m=1, axis=0):
    """
    Computes a single velocity dispersion profile for a set of DM particles
    about a halo center.

    Parameters
    ----------
    halo_center: 3*1 float array
        Halo center position
    dm_part_pos: Ndm*3 float array
        Positions of all relevant dm particles
    dm_part_pos: Ndm*3 float array
        Velocities of all relevant dm particles
    bins: Nbins*1 float array
        Array containing edges of radial bins for profile calculations
    boxsize    : float
        Boxsize for periodic BC calculations
    R200m      : float
        Optional float containing R200m for rescaling in bins of r/R200m

    axis.      : int
        Optional integer of 0, 1, or 2 representing x, y, or z as line
        of site axis.

    Returns
    -------
    avg_r: Nbin*1 array
        Average radial position of each density bin
    avg_prof: Nbin*1 array
        Average profile value in each radial bin
    """


    # Speed up calculation by excluding box around max radius
    delta_x = np.abs(dm_part_pos.T[0]-halo_center[0])
    delta_y = np.abs(dm_part_pos.T[1]-halo_center[1])
    delta_z = np.abs(dm_part_pos.T[2]-halo_center[2])

    R_max = R200m*np.max(bins)

    box_filt = np.where(((delta_x <= R_max) | (boxsize-delta_x <= R_max)) &
                        ((delta_y <= R_max) | (boxsize-delta_y <= R_max)) &
                        ((delta_z <= R_max) | (boxsize-delta_z <= R_max)))

    dm_part_pos = dm_part_pos[box_filt]
    dm_part_vel = dm_part_vel[box_filt]

    # Account for periodic bc
    dev = dm_part_pos-halo_center

    for ind, q in enumerate(dev.T):
        q = np.where(np.abs(q) > 0.5 * boxsize, boxsize-np.abs(q), q)
        dev.T[ind] = q

    # Compute and return profile
    r = np.abs(dev.T[axis])/R200m
    vel = dm_part_vel.T[axis]
    weights = (vel-np.mean(vel))**2

    sigma_sq, avg_r, _ = binned_statistic(r, weights, "mean", bins)
    sigma = np.sqrt(sigma_sq)
    avg_r = 1/2*(avg_r[1:]+avg_r[0:-1])

    return avg_r, sigma


def compute_rad_veloc_prof(halo_center, dm_part_pos, dm_part_vel, bins, boxsize,
                           R200m=1, bin_stat="mean"):
    """
    Compute a single radial velocity profile.

    Parameters
    ----------
    halo_center: 3*1 float array
        Halo center position
    dm_part_pos: Ndm*3 float array
        Positions of all relevant dm particles
    dm_part_vel: Ndm*3 float array
        Velocities of all relevant dm particles
    bins: Nbins*1 float array
        Array containing edges of radial bins for profile calculations
    boxsize    : float
        Boxsize for periodic BC calculations
    R200m      : float
        Optional float containing R200m for rescaling in bins of r/R200m
    bin_stat   : str
        Optional string specifying whether to compute mean or median profile

    Returns
    -------
    avg_r: Nbin*1 array
        Average radial position of each density bin
    avg_prof: Nbin*1 array
        Aveage profile value in each radial bin
    """


    # Speed up calculation by excluding box around max radius
    delta_x = np.abs(dm_part_pos.T[0]-halo_center[0])
    delta_y = np.abs(dm_part_pos.T[1]-halo_center[1])
    delta_z = np.abs(dm_part_pos.T[2]-halo_center[2])

    R_max = R200m*np.max(bins)

    box_filt = np.where(((delta_x <= R_max) | (boxsize-delta_x <= R_max)) &
                        ((delta_y <= R_max) | (boxsize-delta_y <= R_max)) &
                        ((delta_z <= R_max) | (boxsize-delta_z <= R_max)))

    dm_part_pos = dm_part_pos[box_filt]
    dm_part_vel = dm_part_vel[box_filt]

    # Account for periodic bc
    dev = dm_part_pos-halo_center

    for ind, q in enumerate(dev.T):
        q = np.where(np.abs(q) > 0.5 * boxsize, boxsize-np.abs(q), q)
        dev.T[ind] = q

    # Compute radius and radial velocity
    r       = np.linalg.norm(dev, axis=1)/R200m
    r[r==0] = 1e-9 # Don't divide by zero for most bound particle (where r=0)
    r_hat   = dev*R200m/r[:, np.newaxis]
    v_rad   = np.sum(dm_part_vel*r_hat, axis=1)

    # Compute profile
    avg_prof, avg_r, _ = binned_statistic(r, v_rad, bin_stat, bins)
    avg_r = 1/2*(avg_r[1:]+avg_r[0:-1])

    return avg_r, avg_prof


def compute_avg_profiles(halo_centers, dm_part_pos, bins, boxsize, weights=None,
                         R200m_arr=None, bin_stat="mean", return_mat=True,
                         prof_type="dens", axis=0, dm_part_vel=None):

    """
    Compute averaged profiles about a set of halo centers

    Parameters
    ----------
    halo_centers: Nhalo*1 float array
        Array containg centers of desired halos to analyze
    dm_part_pos : Ndm*3 float array
        Positions of all relevant dm particles
    bins        : Nbin*1 float array
        Array containing edges of radial bins for profile calculations
    boxsize     : float
        Boxsize for periodic BC calculations
    weights     : Ndm*1 float Array
        Optional array containing weights for each particle (e.g. for volume
        or mass weighted quantities). Default is None
    R200m_arr   : Nhalo*1 float array
        Optional array containing R200m values of each halo for stacking in
        units of r/R200m.
    bin_stat   : str
        String specifying the statistic you want to apply when stacking over
        cluster. Default is "mean".
    prof_type   : str
        Optional boolean specifying the profile type. Current support options
        are "dens", "weighted", and "vel_disp".
    axis.       : int
        Optional integer of 0, 1, or 2 representing x, y, or z as line
        of site axis. Used only when computing velocity dispersion profiles


    Returns
    -------
    avg_r  : Nbin*1 array
        Average radial position of each bin
    avr_rho: Nbin*1 array
        Average stacked density in each radial bin
    cov    : Nbin*Nbin array
        Jackknife covariance estimate of density profiles
    """

    implemented_profs = ["dens", "weighted", "vel_disp", "rad_veloc"]

    assert prof_type in implemented_profs, \
        (TermColors.CRED+"Profile type specifier must be\n->"+'\n->'.join(str(p) for p in implemented_profs)+
         TermColors.CEND)

    print("\n\nComputing stacked "+prof_type+" profiles for {} halos:".format(len(halo_centers)))

    N_halo = len(halo_centers)
    n_bins = len(bins)-1

    if R200m_arr is None:
        print("-> profiles will be binned in units of r")
        R200m_arr = np.ones(N_halo, dtype=float)
    else:
        print("-> profiles will be binned in units of r/R200m ")
        assert len(R200m_arr)==N_halo, "Size of R200m arr not equal to # of halos"

    if weights is None:
        print("-> no weights specified -- assuming equal particle weighting.")
        weights = np.ones(len(dm_part_pos))
    elif isinstance(weights, (int, float)):
        print("-> single number specified for profile weighting.")
        weights = weights*np.ones(len(dm_part_pos))


    def compute_profile_ind(ind):
        """
        Helper function to compute the profile given the index of the halo
        center (functions allows for parallelization)
        """
        # Compute and store profiles
        if prof_type=="dens":
            r, rho = compute_density_profile(halo_centers[ind], dm_part_pos, bins,
                                             boxsize, weights, R200m_arr[ind])
        elif prof_type=="weighted":
            r, rho = compute_weighted_profile(halo_centers[ind], dm_part_pos, bins,
                                             boxsize, weights, R200m_arr[ind], bin_stat)
        elif prof_type=="vel_disp":
            r, rho = compute_veloc_disp_prof(halo_centers[ind], dm_part_pos, dm_part_vel, bins,
                                             boxsize, R200m_arr[ind], axis)

        elif prof_type=="rad_veloc":
            r, rho = compute_rad_veloc_prof(halo_centers[ind], dm_part_pos, dm_part_vel, bins,
                                            boxsize, R200m_arr[ind], bin_stat)

        return r, rho

    # Compute profiles and format output
    par_res = joblib.Parallel(n_threads)(joblib.delayed(compute_profile_ind)(i)
                                         for i in tqdm(range(N_halo)))

    par_res = np.array(par_res)
    r_mat = par_res[:,0,:]
    rho_mat = par_res[:,1,:]

    # Compute average profile
    print("Individual profiles computed. Now computing averaged profile!")
    r_full = r_mat.flatten()
    rho_full = rho_mat.flatten()

    avg_rho, avg_r, _ = binned_statistic(r_full, rho_full, bin_stat, bins)
    avg_r = 1/2*(avg_r[1:]+avg_r[0:-1])

    # Remove any bins which don't have any points
    non_zero_indices = np.where(avg_rho != 0)
    avg_rho = avg_rho[non_zero_indices]
    avg_r = avg_r[non_zero_indices]

    # Compute covariance
    cov = np.zeros((n_bins, n_bins))

    for k in range(N_halo):
        rho_k = rho_mat[k]

        for i in range(n_bins):
            for j in range(n_bins):
                cov[i][j] += (rho_k[i]-avg_rho[i])*(rho_k[j]-avg_rho[j])

    cov /= N_halo*(N_halo-1)

    if return_mat:
        return {'r': avg_r, 'rho': avg_rho, 'cov': cov, 'rho_mat': rho_mat}
    else:
        return {'r': avg_r, 'rho': avg_rho, 'cov': cov}


