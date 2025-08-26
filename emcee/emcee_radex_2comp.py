#!/usr/bin/env python
# To run it on the cluster, login and use
#
# > srun -N 1 -c 16 --exclusive emcee_radex_2comp.py
#
# to launch it on a node with 16 core, or use
#
# > sbatch emcee_radex_2comp.py
#
#
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --partition=cpu_only
#SBATCH --account=cyang
#SBATCH --exclusive
#SBATCH --mail-type=END

# autopep8 --ignore E26 emcee_radex_2comp.py
import os
# For runing the code on the clusters
import sys
import logging
import _pickle as pickle
import warnings
import numpy as np
from astropy.io import ascii
from astropy import units as u
from astropy.table import Table
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d

# to match the cosmo in the CO paper
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Om0=0.308)

import matplotlib
# Define the fonts to make plots look consistent across different machines
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Linux Biolinum'
matplotlib.rcParams['mathtext.it'] = 'Linux Biolinum:italic'
matplotlib.rcParams['mathtext.bf'] = 'Linux Biolinum:bold'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import multiprocessing
import emcee
import corner
import pyradex
# import pyradex.fjdu


kms = u.km / u.s
Jykms = u.Jy * kms

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ncpu = multiprocessing.cpu_count()

#Assuming ortho to para ratio equals 3
opr    = 3
fortho = opr/(1+opr)


def model_lvg(Jup, p, R=None):
    # component 1  +  component 2
    log_density_1, log_temperature_1, log_column_1, log_size_1, \
    log_density_2, log_temperature_2, log_column_2, log_size_2 = p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ## 1st component
        R.set_params(density={'oH2':fortho*10.**log_density_1,'pH2':(1-fortho)*10.**log_density_1},
                     column=10.**log_column_1,
                     temperature=10.**log_temperature_1)
        R.run_radex(validate_colliders=False,
                    reuse_last=True, reload_molfile=False)
        result_1 = R.source_line_surfbrightness # Do not use get_table()
        ## 2nd component
        R.set_params(density={'oH2':fortho*10.**log_density_2,'pH2':(1-fortho)*10.**log_density_2},
                     column=10.**log_column_2,
                     temperature=10.**log_temperature_2)
        R.run_radex(validate_colliders=False,
                    reuse_last=True, reload_molfile=False)
        result_2 = R.source_line_surfbrightness # Do not use get_table()

    intensity = (result_1[np.asarray(np.int_(Jup)) - 1] * (10.**log_size_1 * u.sr) * (1. * kms)).to(Jykms) + \
                (result_2[np.asarray(np.int_(Jup)) - 1] * (10.**log_size_2 * u.sr) * (1. * kms)).to(Jykms)
    return intensity.value


def model_single_lvg(Jup, p, R=None):
    log_density, log_temperature, log_column, log_size = p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R.set_params(density={'oH2':fortho*10.**log_density,'pH2':(1-fortho)*10.**log_density},
                     column=10.**log_column,
                     temperature=10.**log_temperature)
        R.run_radex(validate_colliders=False,
                    reuse_last=True, reload_molfile=False)
        result = R.source_line_surfbrightness # Do not use get_table()
    intensity = (result[np.asarray(np.int_(Jup)) - 1] *
                 (10.**log_size * u.sr) * (1. * kms)).to(Jykms)
    return intensity.value


def model_lvg_tau(p, R=None):
    # component 1  +  component 2
    log_density_1, log_temperature_1, log_column_1, log_size_1, \
    log_density_2, log_temperature_2, log_column_2, log_size_2 = p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ## 1st component 
        R.set_params(density={'oH2':fortho*10.**log_density_1,'pH2':(1-fortho)*10.**log_density_1},
                     column=10.**log_column_1,
                     temperature=10.**log_temperature_1)
        R.run_radex(validate_colliders=False,
                    reuse_last=True, reload_molfile=False)
        result_1 = np.amax(R.tau)
        ## 2nd component 
        R.set_params(density={'oH2':fortho*10.**log_density_2,'pH2':(1-fortho)*10.**log_density_2},
                     column=10.**log_column_2,
                     temperature=10.**log_temperature_2)
        R.run_radex(validate_colliders=False,
                    reuse_last=True, reload_molfile=False)
        result_2 = np.amax(R.tau)
    return np.amax([result_1,result_2])


def residual(p, R=None, Jup=None, flux=None, eflux=None):
    model_flux = model_lvg(Jup, p, R)
    return (flux - model_flux) / eflux


def lnlike(p, Jup, flux, eflux, R=None):
    """ likelihood function"""
    try:
        model_flux = model_lvg(Jup, p, R)
    except ValueError:
        return -np.inf
    # Some nan sometimes....
    if np.any(np.isnan(model_flux)):
        return -np.inf
    return -0.5 * (np.sum(((flux - model_flux)**2.0 / eflux**2.0)+np.log(eflux**2.0)))


def lnprior(p, bounds, T_d=None, R=None):
    """Gaussian prior(T_d)"""
    #     n_H2     T_kin    N_CO    size
    #     p[0]     p[1]     p[2]    p[3]   ## 1st component
    #     p[4]     p[5]     p[6]    p[7]   ## 2nd component
    log_density_1, log_temperature_1, log_column_1, log_size_1,\
    log_density_2, log_temperature_2, log_column_2, log_size_2 = p

    # First Check boundaries
    if (np.any(p > bounds[:, 1]) or np.any(p < bounds[:, 0])):
        return -np.inf

    # Forcing the 2nd component to be warmer
    if (np.any(p[5] <= p[1])):
        return -np.inf

    # 9 < log10(N_CO/dv) - log10(n_H2) < 18
    if (np.any((p[2] - p[0]) >= 18.0) or np.any((p[2] - p[0]) <= 9.0) or
        np.any((p[6] - p[4]) >= 18.0) or np.any((p[6] - p[4]) <= 9.0)) :
        return -np.inf

    ## The size of the cold component should be larger!
    if (np.any((p[3] < p[7]))):
        return -np.inf

    # Add bounds for tau, should be < 100
    # try:
    #     tau_max = model_lvg_tau(p, R)
    # except ValueError:
    #     return -np.inf
    # 
    # if (tau_max > 100):
    #     return -np.inf

    logp=0
    for index, (value, bound) in enumerate(zip(p, bounds)):
        if index == 1:
            # First T_kin -> Gaussian around (T_d, sigma_T_d=1*T_d)
            logp += (-((10.0**value-T_d)/(1.0*T_d))**2.0/2.0 - np.log(T_d*np.sqrt(2.0*np.pi)))
        else:
            # Classical Uniform prior
            logp += -(bound[1]-bound[0])
    return logp


def lnprob(p, Jup, flux, eflux, bounds=None, T_d=None):
    lp = lnprior(p, bounds, T_d=T_d, R=R)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p, Jup, flux, eflux, R=R)


def read_data(filename):
    """Read data into a comprehensible panda frame"""

    ######################################################
    #  Disadvantage here: only includes J_up = 11 here,  #
    #  please manually add more if you have              #
    #  J_up >= 12 CO lines                               #
    ######################################################

    ascii_data = ascii.read(
        filename, names=[
            "SOURCE", "z", "D_L", "T_d", "line_width",
            "CO_J_1", "eCO_J_1", "CO_J_2", "eCO_J_2", "CO_J_3", "eCO_J_3",
            "CO_J_4", "eCO_J_4", "CO_J_5", "eCO_J_5", "CO_J_6", "eCO_J_6",
            "CO_J_7", "eCO_J_7", "CO_J_8", "eCO_J_8", "CO_J_9", "eCO_J_9",
            "CO_J_10", "eCO_J_10", "CO_J_11", "eCO_J_11", "CI_1", "eCI_1",
            "CI_2", "eCI_2"])

    pd = ascii_data.to_pandas()
    pd = pd.set_index('SOURCE')
    return pd.T


def get_source(source, data):
    """Retrieve redshift and CO data for a particular source"""

    # source = 'NCv1.143'
    z = data[source]['z']
    line_width = data[source]['line_width']
    T_d     = data[source]['T_d']
    delta_v = 1 * kms  # do not care actually, fully degenerate with
                       # the column density

    # selecting only CO lines
    keys = [key for key in data[source].keys()
            if 'CO' in key and 'eCO' not in key]
    CO_data = Table(np.asarray([(Jlow + 1, data[source][key], data[source]['e' + key])
                                for Jlow, key in enumerate(keys)
                                if np.isfinite(data[source][key])]),
                    names=['Jup', 'flux', 'eflux'],
                    dtype=[int, float, float])

    Jup = CO_data['Jup'].data
    flux = CO_data['flux'].data * Jykms
    eflux = CO_data['eflux'].data * Jykms

    return z, T_d, line_width, Jup, flux, eflux


# the pyradex.Radex object MUST be declared at the module level for
# lnprob to work properly
#  ....  Setup the Radex parameters

# Note that N_CO is totaly degenerated with deltav, so in
# principle we should fit for N_CO/deltav
R = pyradex.Radex(species='co', datapath="radex_moldata",
                  density={'oH2':fortho*10.**10.0,'pH2':(1-fortho)*10.**10.0},
                  column=10.**6.0,
                  temperature=20.0,
                  tbackground=2.7315,
                  deltav=1.0,
                  escapeProbGeom='lvg')

# R = pyradex.fjdu.Fjdu(species='co', datapath="radex_moldata",
#                   density={'oH2':fortho*10.**10,'pH2':(1-fortho)*10.**10},
#                   column=10**6,
#                   temperature=20,
#                   tbg=2.7315,
#                   deltav=1,
#                   escapeProbGeom='lvg')


def replot(source):

    plt.ion()
    # Retrieve the data
    with open("./double/{}_bounds_2comp.pickle".format(source), 'rb') as pkl_file:
        (source, z, bounds, T_d,
            (Jup, flux, eflux), (popt, pcov), pmin, pemcee, (chain, lnprobability)) =  pickle.load(pkl_file)

    R.set_params(tbg=2.7315 * (1 + z))

    # Get the max posterior within +/-1 sigma range
    flatchain = chain.reshape((chain.shape[0]*chain.shape[1]),8)
    lnp = lnprobability.reshape((lnprobability.shape[0]*lnprobability.shape[1]),1)
    lower, upper     = np.percentile(flatchain, [16, 84],axis=0)
    narrow_flatchain = flatchain[(flatchain[:,0] > lower[0]*1) & (flatchain[:,0] < upper[0]*1) & \
                                 (flatchain[:,1] > lower[1]*1) & (flatchain[:,1] < upper[1]*1) & \
                                 (flatchain[:,2] > lower[2]*1) & (flatchain[:,2] < upper[2]*1) & \
                                 (flatchain[:,3] > lower[3]*1) & (flatchain[:,3] < upper[3]*1) & \
                                 (flatchain[:,4] > lower[4]*1) & (flatchain[:,4] < upper[4]*1) & \
                                 (flatchain[:,5] > lower[5]*1) & (flatchain[:,5] < upper[5]*1) & \
                                 (flatchain[:,6] > lower[6]*1) & (flatchain[:,6] < upper[6]*1) & \
                                 (flatchain[:,7] > lower[7]*1) & (flatchain[:,7] < upper[7]*1) ]
    narrow_lnp       =       lnp[(flatchain[:,0] > lower[0]*1) & (flatchain[:,0] < upper[0]*1) & \
                                 (flatchain[:,1] > lower[1]*1) & (flatchain[:,1] < upper[1]*1) & \
                                 (flatchain[:,2] > lower[2]*1) & (flatchain[:,2] < upper[2]*1) & \
                                 (flatchain[:,3] > lower[3]*1) & (flatchain[:,3] < upper[3]*1) & \
                                 (flatchain[:,4] > lower[4]*1) & (flatchain[:,4] < upper[4]*1) & \
                                 (flatchain[:,5] > lower[5]*1) & (flatchain[:,5] < upper[5]*1) & \
                                 (flatchain[:,6] > lower[6]*1) & (flatchain[:,6] < upper[6]*1) & \
                                 (flatchain[:,7] > lower[7]*1) & (flatchain[:,7] < upper[7]*1) ]  
    pemcee_max       = narrow_flatchain[narrow_lnp.argmax()]

    model_Jup = range(1, 12)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    minorLocator_x = MultipleLocator(1)
    #minorLocator_y = MultipleLocator(0.5)
    ax.errorbar(Jup, flux.value, eflux.value, fmt='o', label=r'$\mathrm{data}$', color='#000000', capsize=0)
    plot_Jup = np.arange(min(model_Jup), max(model_Jup), 0.05) # smoothing the model line

    pemcee_max_c = pemcee_max[:4]
    pemcee_max_w = pemcee_max[4:]
    pemcee_c = pemcee[:4]
    pemcee_w = pemcee[4:]

    f_inter_pemcee_max   = interp1d(model_Jup,        model_lvg(model_Jup, pemcee_max,   R), kind='cubic')
    f_inter_pemcee_max_c = interp1d(model_Jup, model_single_lvg(model_Jup, pemcee_max_c, R), kind='cubic')
    f_inter_pemcee_max_w = interp1d(model_Jup, model_single_lvg(model_Jup, pemcee_max_w, R), kind='cubic')
    f_inter_pemcee       = interp1d(model_Jup,        model_lvg(model_Jup, pemcee,       R), kind='cubic')
    f_inter_pemcee_c     = interp1d(model_Jup, model_single_lvg(model_Jup, pemcee_c,     R), kind='cubic')
    f_inter_pemcee_w     = interp1d(model_Jup, model_single_lvg(model_Jup, pemcee_w,     R), kind='cubic')

    ax.plot(plot_Jup,   f_inter_pemcee_max(plot_Jup), label=r'$\mathrm{{MCMC}}$', color='#FFA833')
    ax.plot(plot_Jup, f_inter_pemcee_max_w(plot_Jup), linestyle='--', color='#fcc82d')
    ax.plot(plot_Jup, f_inter_pemcee_max_c(plot_Jup), linestyle='-.', color='#ff7b33')

    #ax.plot(plot_Jup,       f_inter_pemin(plot_Jup), label=r'$\mathrm{median_{MCMC}}$', color='#58b82a')
    #ax.plot(plot_Jup,     f_inter_pemcee_c(plot_Jup), linestyle='--', color='#198189')
    #ax.plot(plot_Jup,     f_inter_pemcee_w(plot_Jup), linestyle=':', color='#b1d623')

    # Plot the 200 "good models" within the [16, 84] quartile
    inds = np.random.randint(len(narrow_flatchain), size=200)
    for ind in inds:
        sample = narrow_flatchain[ind]
        model_flux = model_lvg(model_Jup, sample, R)
        f_inter_pemcee_sample = interp1d(model_Jup, model_flux, kind='cubic')
        ax.plot(plot_Jup, f_inter_pemcee_sample(plot_Jup), color='#f5ec42', alpha=0.1)

    ax.set_xlabel(r'$J_\mathrm{up}$',fontsize=14)
    ax.set_ylabel(r'$I_\mathrm{CO}\;[\mathrm{Jy\;km\;s^{-1}}]$',fontsize=14)
    ax.legend(loc=0, prop={'size':11}, numpoints=1)
    ax.xaxis.set_minor_locator(minorLocator_x)
    fig.suptitle(r'$\mathrm{'+source+'}$',fontsize = 15)
    fig.savefig("./double/{}_SLED_2comp.pdf".format(source))

    # plots for the full corner
    chain_plot = np.hstack((flatchain[:,[0,1,2,3]], flatchain[:,[4,5,6,7]]))
    plot_range=[(1.9,7.1),(1,3.02),(14.5, 19.5),(-12.5,-8.5),(1.9,7.1),(1,3.0),(14.5, 19.5),(-12.5,-8.5)]
    fig = corner.corner(chain_plot,
                labels=[r'$\mathrm{log}_{10}(n_\mathrm{H_2,\,c}\;[\mathrm{cm}^{-3}])$',
                        r'$\mathrm{log}_{10}(T_\mathrm{kin,\,c}\;[\mathrm{K}])$',
                        r'$\mathrm{log}_{10}({N_\mathrm{CO,\,c}}/{\mathrm{d}v}\;[\frac{\mathrm{cm}^{-2}}{\mathrm{km\,s}^{-1}}])$',
                        r'$\mathrm{log}_{10}(\mathrm{[size,\,c\,sr^{-1}]})$',
                        r'$\mathrm{log}_{10}(n_\mathrm{H_2,\,w}\;[\mathrm{cm}^{-3}])$',
                        r'$\mathrm{log}_{10}(T_\mathrm{kin,\,w}\;[\mathrm{K}])$',
                        r'$\mathrm{log}_{10}({N_\mathrm{CO,\,w}}/{\mathrm{d}v}\;[\frac{\mathrm{cm}^{-2}}{\mathrm{km\,s}^{-1}}])$',
                        r'$\mathrm{log}_{10}(\mathrm{[size,\,w\,sr^{-1}]})$',
                        ],
                show_titles=True, title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 15},
                plot_datapoints=False, range=plot_range, max_n_ticks=6, smooth=0.8,
                quantiles=[0.15865, 0.50, 0.84135], truths=np.hstack((pemcee_max_c,pemcee_max_w)), 
                truth_color="#FFA833", color="#2B61DD", bins=24)
    fig.suptitle(r'$\mathrm{'+source+'}$',fontsize = 16)
    fig.savefig("./double/{}_corner_2comp_all.pdf".format(source))

    # plots for publication, remove size from the plot
    chain_cold = flatchain[:,[0,1,2]]
    chain_warm = flatchain[:,[4,5,6]]
    new_pemcee_max_c = np.hstack((pemcee_max_c[:3]))  # only show n_H2, T_kin, N_CO and Pressure
    new_pemcee_max_w = np.hstack((pemcee_max_w[:3]))  # only show n_H2, T_kin, N_CO and Pressure
    plot_range=[(1.9,7.1),(1,3.02),(14.5, 19.5)]
    fig = corner.corner(chain_cold,
                        labels=[r'$\mathrm{log}_{10}(n_\mathrm{H_2}\;[\mathrm{cm}^{-3}])$',
                                r'$\mathrm{log}_{10}(T_\mathrm{kin}\;[\mathrm{K}])$',
                                r'$\mathrm{log}_{10}({N_\mathrm{CO}}/{\mathrm{d}v}\;[\frac{\mathrm{cm}^{-2}}{\mathrm{km\,s}^{-1}}])$'],
                                show_titles=True, title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 15}, 
                                plot_datapoints=False, range=plot_range, max_n_ticks=6, smooth=0.8,
                                quantiles=[0.15865, 0.50, 0.84135], truths=new_pemcee_max_c,
                                truth_color="#fcc82d", color="#198189", bins=24)
    fig.savefig("./double/{}_corner_2comp_1.pdf".format(source))

    fig = corner.corner(chain_warm,
                        labels=[r'$\mathrm{log}_{10}(n_\mathrm{H_2}\;[\mathrm{cm}^{-3}])$',
                                r'$\mathrm{log}_{10}(T_\mathrm{kin}\;[\mathrm{K}])$',
                                r'$\mathrm{log}_{10}({N_\mathrm{CO}}/{\mathrm{d}v}\;[\frac{\mathrm{cm}^{-2}}{\mathrm{km\,s}^{-1}}])$'],
                                show_titles=True, title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 15}, 
                                plot_datapoints=False, range=plot_range, max_n_ticks=6, smooth=0.8,
                                quantiles=[0.15865, 0.50, 0.84135], truths=new_pemcee_max_w,
                                truth_color="#ff7b33", color="#b1d623", bins=24)
    fig.savefig("./double/{}_corner_2comp_2.pdf".format(source))

    # Print the MCMC results
    chain_plot_cold = np.hstack((chain_cold[:,[0,1,2]], chain_cold[:,[0]]+chain_cold[:,[1]]))
    chain_plot_warm = np.hstack((chain_warm[:,[0,1,2]], chain_warm[:,[0]]+chain_warm[:,[1]]))
    n_c, T_c, N_c, P_c = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                         list(zip(*np.percentile(chain_plot_cold, [16, 50, 84], axis=0))))
    n_w, T_w, N_w, P_w = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                         list(zip(*np.percentile(chain_plot_warm, [16, 50, 84], axis=0))))

    print("#### cold component - median ####")
    print(' ', n_c[0],' ', T_c[0],' ', N_c[0],' ', P_c[0])
    print('+', n_c[1],'+', T_c[1],'+', N_c[1],'+', P_c[1])
    print('-', n_c[2],'-', T_c[2],'-', N_c[2],'-', P_c[2])
    print("4max", '\n', pemcee_max_c)
    print("=================================")
    print("#### warm component - median ####")
    print(' ', n_w[0],' ', T_w[0],' ', N_w[0],' ', P_w[0])
    print('+', n_w[1],'+', T_w[1],'+', N_w[1],'+', P_w[1])
    print('-', n_w[2],'-', T_w[2],'-', N_w[2],'-', P_w[2])
    print("4max", '\n', pemcee_max_w)
    print("=================================")


def main():

    data = read_data("../data/flux_for2p.dat")

    if not os.path.exists("./double"):
        os.makedirs("./double")

    for source in data.keys():
    # if True:
    #     source = "SDP81"
    #     source = 'NCv1.143'

        logger.info('Processing {}'.format(source))

        delta_v = 1 * kms  # do not care actually, fully degenerate
                           # with the column density

        # TODO: would it help to use the CI data ?
        # If 2 lines probably yes, otherwise, must be degenerate with
        # the abondance

        # Retrieve data
        z, T_d, line_width, Jup, flux, eflux = get_source(source, data)
        R.set_params(tbg=2.7315 * (1 + z))

        # Line : Jy km/s
        # 1 Jansky=1 Jy = 10e-26 W m-2 Hz-1 = 1e-23 erg s-1 cm-2 Hz-1
        # brightness erg s-1 cm-2 Hz-1 sr-1 ~ W m-2 Hz-1 sr-1 ~ Jy sr-1

        # Radex output : erg s-1 cm-2  = 1e3 W m-2
        # pyradex output : erg s-1 cm-2 Hz-1 sr-1

        # K*km/s is an integrated surface brightness, while the value
        # in erg/s/cm2 is an isotopic flux emitted in all directions.

        # Specicic Intensity  I(nu)  W m-2 Hz-1 sr-1
        # Flux density W m-2 Hz-1 ~ 10e26 Jy : S = I * Surf./d**2

        # -> Specific (velocity integrated ) Intensity
        # I = S_nu * d**2 / surf

        # Size:
        # R_source = D_A * sqrt(size/pi)
        # This is uncorrected for lensing magification

        # H2 density : Number density of collision partners : Unit: cm-3.  ; Allowed range: 10(-3) - 10(13)
        # Column density : Unit: cm-2 : Allowed range: 10(5) - 10(25)
        # Kinetic temperature : Unit: K. Allowed range: 0.1 - 10,000
        # linewidth : Thermal + turbulent width (FWHM) of the lines.; Unit: km s-1. ; Allowed range: 10(-3) - 10(3)
        # Size : Unit sr

        bounds = np.array([[-3, 13],    # log H2 density
                           [2, 3000],   # Kinetic temperature
                                        # (Collisions temperature of CO
                                        # in radex files)
                           [5, 25],     # log Column density
                           [-20, -5],   # log size
                           ##############
                           [-3, 13],    # log H2 density
                           [2, 3000],   # Kinetic temperature
                                        # (Collisions temperature of CO
                                        # in radex files)
                           [5, 25],     # log Column density
                           [-20, -5]])  # log size

        R_angle = ((7/(cosmo.angular_diameter_distance(z).value*1000.0))**2 * np.pi)*10
        # Assuming, 7 kpc size and mu=10 lensing magnification

        # basic starting point
        p0 = [5.6,     # np.log10(total_density)
              24,      # temperature
              18,      # 10**16.75*300, # np.log10(column)
              np.log10(R_angle),  # np.log10(size)
              ##########
              5.6,     # np.log10(total_density)
              24,      # temperature
              18,      # 10**16.75*300, # np.log10(column)
              np.log10(R_angle)]  # np.log10(size)

        # As suggested by https://arxiv.org/pdf/1212.5955.pdf
        # Further narrowing down according to Zhang et al, A&A 568, A122 (2014)
        # remove unrealistic ranges
        # Ranges:
        # n_H2    = 10^1.5 -- 10^7.0 cm^-3
        # T_kin   = T_CMB -- 10^3 K
        # N_CO/dv = 10^15.5 -- 10^19.5 cm^-2 (km/s)^-1
        # dv/dr   = 0.1 -- 1000 (Tunnard+2016, Xco=5e-5), saying r ~ 1-5 kpc, delta_V = 250-700 km/s
        #  -->   6.18e13 < N_CO/n_H2 < 6.18e17
        #  -->   due to lensing uncertainties, add +/- 50 lensing uncertainty factor, multiply delta_V = delta_V = 250-700 km/s
        #  -->   10.0 < log10(N_CO/dv) - log10(n_H2) < 17.5,
        # Additional constrains:
        # N_CO/(n_H2*Xco) < 2 R = 10 kpc (assuming disk is =<5 kpc, Xco=5e-5)
        #  -->   N_CO/n_H2 < 5e19, within the dv/dr range.
                           ##### 1st component #####
        bounds = np.array([[1.5, 7.0],
                           [np.log10(2.7315 * (1 + z)), 3.0],
                           [14.5, 19.5],
                           [np.log10(R_angle)-5, np.log10(R_angle)+5],
                           ##### 2nd component #####
                           [1.5, 7.0],
                           [np.log10(2.7315 * (1 + z)), 3.0],
                           [14.5, 19.5],
                           [np.log10(R_angle)-5, np.log10(R_angle)+5]])

        p0 = [4.6,      # np.log10(n_H2_cold) = 3.0
              1.5,      # np.log10(T_kin_cold) = 1.5
             17.7,      # 10**16.8*300, # np.log10(column)
            -10.5,      # np.log10(size)
              ###########
              3.8,      # np.log10(n_H2_cold) = 2.9
              2.3,      # nnp.log10(T_kin_warm) = 2.5
             17.6,      # 10**16.7*300, # np.log10(column)
            -11.1]      # np.log10(size)

        # Simple curve_fit to find a starting point
        # Using lambda to get the R object through
        opt_fun = lambda p,  log_density_1, log_temperature_1, log_column_1, log_size_1, \
                             log_density_2, log_temperature_2, log_column_2, log_size_2: \
               model_lvg(p, [log_density_1, log_temperature_1, log_column_1, log_size_1,
                             log_density_2, log_temperature_2, log_column_2, log_size_2], R=R)

        try:
            popt, pcov = curve_fit(opt_fun, Jup, flux.value,
                                   sigma=eflux.value, p0=p0,
                                  bounds=list(zip(*bounds)))
            logger.info("    curve_fit : {}".format(popt))
        except RuntimeError:
            logger.warn("    curve_fit : failed")
            popt = p0
            pcov = None

        # minimize do not work well far from the solution, so use the
        # curve_fit solution to test the lnprob function...
        p = popt
        nll = lambda p, Jup, flux, eflux, : - \
            lnprob(p, Jup, flux, eflux, bounds, T_d)
        result = minimize(nll, p,
                          args=(Jup, flux.value, eflux.value),
                          bounds=bounds)
        pmin = result.x
        logger.info("    minimize : {}".format(pmin))

        # Do the heavy computation
        ndim = len(popt)

        #################### Define the number of walkers here
        nwalkers = 400       # 400 walkers
        n_iter_burn = 100    # burning phase, number of iterations = 100
        n_iter_walk = 1000   # walking phase, number of iterations = 1000

        # Random starting positions
        pos = [popt + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]

        # Multithread
        with multiprocessing.Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                            args=(Jup, flux.value, eflux.value),
                                            kwargs={'bounds': bounds, 'T_d': T_d},
                                            threads=ncpu)
            # Burning time
            logger.info("    burning samples")
            pos, prob, state = sampler.run_mcmc(
                pos, n_iter_burn)  # now 100, will be 1000
            sampler.reset()
            # Sampling time
            logger.info("    walking")
            result = sampler.run_mcmc(pos, n_iter_walk)  # now 1000, will be 5000
            pemcee = np.percentile(sampler.flatchain, [50], axis=0)[0]

        chain, lnprobability = sampler.chain, sampler.lnprobability

        with open("./double/{}_bounds_2comp.pickle".format(source), 'wb') as pkl_file:
            pickle.dump((source, z, bounds, T_d,
                         (Jup, flux, eflux), (popt, pcov), pmin, pemcee, (chain, lnprobability)),
                        pkl_file)

        # Quick plot the model
        # replot(source)

        # Print only the best fitted n_H2, T_kin, N_CO and Pressure
        pmin_c = pmin[:4]
        pmin_w = pmin[4:]

        chain_cold = np.hstack((sampler.flatchain[:,[0,1,2]],
                                sampler.flatchain[:,[0]]+sampler.flatchain[:,[1]]))
        chain_warm = np.hstack((sampler.flatchain[:,[4,5,6]],
                                sampler.flatchain[:,[4]]+sampler.flatchain[:,[5]]))
        new_pmin_c = np.hstack((pmin_c[:3],pmin_c[0]+pmin_c[1]))
        new_pmin_w = np.hstack((pmin_w[:3],pmin_w[0]+pmin_w[1]))

        n_c, T_c, N_c, P_c = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                list(zip(*np.percentile(chain_cold, [16, 50, 84], axis=0))))
        n_w, T_w, N_w, P_w = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                list(zip(*np.percentile(chain_warm, [16, 50, 84], axis=0))))

        # Output the best fit
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("xxx:", source, '\n', "xxx: minimised results")
        print("xxx:", new_pmin_c, '\n', new_pmin_w)
        # Print the MCMC results for +/- 1 sigma range of the paremeters
        print("xxx: emcee results")
        print("xxx:", "n_H2")
        print("xxx:", n_c, '\n', "xxx:",n_w)
        print("xxx:", "T_kin")
        print("xxx:", T_c, '\n', "xxx:",T_w)
        print("xxx:", "N_CO/dv")
        print("xxx:", N_c, '\n', "xxx:",N_w)
        print("xxx:", "P")
        print("xxx:", P_c, '\n', "xxx:",P_w)

if __name__ == '__main__':

    plt.ion()
    # plt.close('all')
    main()


# See :
# - https://arxiv.org/pdf/1602.01095.pdf
# - https://arxiv.org/pdf/1401.2998.pdf
# - https://arxiv.org/pdf/1212.5955.pdf
# - https://arxiv.org/abs/0809.2337
# plt.figure()
# plt.plot(Jup, model[:8]/model[0])