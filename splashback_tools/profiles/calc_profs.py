#*****************************************************************************
# Calculations routines for halo profiles in simulations
#*****************************************************************************

import numpy as np
import os
from scipy.stats import binned_statistic
from tqdm import tqdm
import joblib

if 'OMP_NUM_THREADS' in os.environ.keys():
    n_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    n_threads = os.cpu_count()


def compute_avg_profiles(halo_centers, dm_part_pos, bins, boxsize, is_2d = False, R200m_arr=None, bin_stat="mean", return_mat=True):
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


    Returns
    -------
    avg_r  : Nbin*1 array
        Average radial position of each bin
    avg_dens: Nbin*1 array
        Average stacked density in each radial bin
    cov    : Nbin*Nbin array
        Jackknife covariance estimate of density profiles
    """

    def get_filtered_positions(halo_center, dm_part_pos, bins, boxsize, R200m):
        """
        Compute the relative position of all nearby halos to the halo center

        Parameters
        ----------
        halo_center: 3*1 float array
            Halo center position
        R200m      : float
            Optional float containing R200m for rescaling in bins of r/R200m

        Returns
        -------
        avg_r: Nbin*1 array
            Average radial position of each density bin
        avg_rho: Nbin*1 array
            Average density in each radial bin
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

        # Account for periodic bc
        res = dm_part_pos-halo_center

        for ind, q in enumerate(res.T):
            q = np.where(np.abs(q) > 0.5 * boxsize, boxsize-np.abs(q), q)
            res.T[ind] = q

        return res
    
    def compute_rho(halo_center, dm_part_pos, bins, boxsize, R200m):
        dev = get_filtered_positions(halo_center, dm_part_pos, bins, boxsize, R200m)

        # Compute and return profile
        r = np.linalg.norm(dev, axis=1)/R200m
        counts, r_edges = np.histogram(r, bins=bins)

        avg_r = 1/2*(r_edges[1:]+r_edges[0:-1])
        dV = 4/3*np.pi*(r_edges[1:]**3-r_edges[0:-1]**3)

        avg_rho = counts/dV*R200m**3

        return avg_r, avg_rho

    def compute_sigma(halo_center, dm_part_pos, bins, boxsize, R200m):
        dev = get_filtered_positions(halo_center, dm_part_pos, bins, boxsize, R200m)[:,:2]

        # Compute and return profile
        r = np.linalg.norm(dev, axis=1)/R200m
        counts, r_edges = np.histogram(r, bins=bins)

        avg_r = 1/2*(r_edges[1:]+r_edges[0:-1])
        dA = np.pi * (r_edges[1:]**2-r_edges[0:-1]**2)

        avg_sigma = counts/dA*R200m**2

        return avg_r, avg_sigma

    if is_2d:
        def compute_profile_ind(ind):
            r, sigma = compute_sigma(halo_centers[ind], dm_part_pos, bins, boxsize, R200m_arr[ind])
            return r, sigma
    else:
        def compute_profile_ind(ind):
            r, rho = compute_rho(halo_centers[ind], dm_part_pos, bins, boxsize, R200m_arr[ind])
            return r, rho

    if R200m_arr is None:
            print("-> profiles will be binned in units of r")
            R200m_arr = np.ones(len(halo_centers), dtype=float)

    N_halo = len(halo_centers)
    n_bins = len(bins)-1

    # Compute profiles and format output
    par_res = joblib.Parallel(n_threads)(joblib.delayed(compute_profile_ind)(i)
                                         for i in tqdm(range(N_halo)))

    par_res = np.array(par_res)
    r_mat = par_res[:,0,:]
    dens_mat = par_res[:,1,:]

    # Compute average profile
    print("Individual profiles computed. Now computing averaged profile!")
    r_full = r_mat.flatten()
    dens_full = dens_mat.flatten()

    avg_dens, avg_r, _ = binned_statistic(r_full, dens_full, bin_stat, bins)
    avg_r = 1/2*(avg_r[1:]+avg_r[0:-1])

    # Remove any bins which don't have any points
    non_zero_indices = np.nonzero(avg_dens)
    avg_dens = avg_dens[non_zero_indices]
    avg_r = avg_r[non_zero_indices]

    # Compute covariance
    cov = np.zeros((n_bins, n_bins))

    for k in range(N_halo):
        dens_k = dens_mat[k]

        for i in range(n_bins):
            for j in range(n_bins):
                cov[i][j] += (dens_k[i]-avg_dens[i])*(dens_k[j]-avg_dens[j])

    cov /= N_halo*(N_halo-1)

    if return_mat:
        return {'r': avg_r, 'dens': avg_dens, 'cov': cov, 'dens_mat': dens_mat}
    else:
        return {'r': avg_r, 'dens': avg_dens, 'cov': cov}




