#!/usr/bin/env python
# To run it on the cluster, log into cluster
# and use
#
# > srun -N 1 -c 16 --exclusive just_4_plot.py
#
# to launch it on a node with 16 core, or use
#
# > sbatch just_4_plot.py
#
#
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --partition=cpu_only
#SBATCH --account=cyang
#SBATCH --exclusive
#SBATCH --mail-type=END
#SBATCH --mail-user=chentao.yang@ias.u-psud.fr

# autopep8 --ignore E26 emcee_radex_2comp_v3.py
import os
# For runing the code on the clusters
import sys
sys.path.insert(1, '/home/cyang/.local/lib/python2.7/site-packages')
import logging
import cPickle as pickle
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

    intensity = (result_1[np.asarray(Jup) - 1] * (10.**log_size_1 * u.sr) * (1. * kms)).to(Jykms) + \
                (result_2[np.asarray(Jup) - 1] * (10.**log_size_2 * u.sr) * (1. * kms)).to(Jykms)
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
    intensity = (result[np.asarray(Jup) - 1] *
                 (10.**log_size * u.sr) * (1. * kms)).to(Jykms)
    return intensity.value


def read_data(filename):
    """Read data into a comprehensible panda frame"""

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


def replot(source):

    plt.ion()

    # Retrieve the single data
    with open("./single/{}_bounds.pickle".format(source), 'r') as pkl_file:
        (source, z, bounds, (Jup, flux, eflux), (popt, pcov), pmin, pemcee, (chain, lnprobability)) =  pickle.load(pkl_file)

    R.set_params(tbg=2.7315 * (1 + z))
    
    # Get the max posterior within +/-1 sigma range
    flatchain = chain.reshape((chain.shape[0]*chain.shape[1]),4) 
    lnp = lnprobability.reshape((lnprobability.shape[0]*lnprobability.shape[1]),1)
    lower, upper     = np.percentile(flatchain, [16, 84],axis=0)
    narrow_flatchain = flatchain[(flatchain[:,0] > lower[0]*1.02) & (flatchain[:,0] < upper[0]*0.98) & \
                                 (flatchain[:,1] > lower[1]*1.02) & (flatchain[:,1] < upper[1]*0.98) & \
                                 (flatchain[:,2] > lower[2]*1.02) & (flatchain[:,2] < upper[2]*0.98) & \
                                 (flatchain[:,3] > lower[3]*1.02) & (flatchain[:,3] < upper[3]*0.98) ]
    narrow_lnp       =       lnp[(flatchain[:,0] > lower[0]*1.02) & (flatchain[:,0] < upper[0]*0.98) & \
                                 (flatchain[:,1] > lower[1]*1.02) & (flatchain[:,1] < upper[1]*0.98) & \
                                 (flatchain[:,2] > lower[2]*1.02) & (flatchain[:,2] < upper[2]*0.98) & \
                                 (flatchain[:,3] > lower[3]*1.02) & (flatchain[:,3] < upper[3]*0.98) ]   
    pemcee_max       = narrow_flatchain[narrow_lnp.argmax()]
    print pemcee_max
    
    model_Jup = range(1, 12)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    minorLocator_x = MultipleLocator(1)
    minorLocator_y = MultipleLocator(0.5)
    ax.errorbar(Jup, flux.value, eflux.value, fmt='o', color='#000000', capsize=0, label=r'$\mathrm{data}$')
    ax.xaxis.set_minor_locator(minorLocator_x)
    ax.yaxis.set_minor_locator(minorLocator_y)
    plot_Jup = np.arange(min(model_Jup), max(model_Jup), 0.05) # smoothing the model line
    f_inter_pemcee_max_single = interp1d(model_Jup, model_single_lvg(model_Jup, pemcee_max, R), kind='cubic')
    ysingle = f_inter_pemcee_max_single(plot_Jup)
    
    
    # plots for publication, remove size from the plot
    plot_range=[(1.9,7.1),(1,3.02),(14.5, 19.5)]
    fig = corner.corner(flatchain[:,[0,1,2]],
          labels=[r'$\mathrm{log}_{10}(n_\mathrm{H_2}\;[\mathrm{cm}^{-3}])$',
                  r'$\mathrm{log}_{10}(T_\mathrm{kin}\;[\mathrm{K}])$',
                  r'$\mathrm{log}_{10}({N_\mathrm{CO}}/{\mathrm{d}v}\;[\frac{\mathrm{cm}^{-2}}{\mathrm{km\,s}^{-1}}])$'],
          show_titles=True, title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 15}, 
          plot_datapoints=False, range=plot_range, max_n_ticks=6, smooth=0.6,
          quantiles=(0.15865, 0.84135), truths=pemcee_max[:3], truth_color="#FFA833", color="#2B61DD", bins=24)
    fig.suptitle('$\mathrm{'+source+'}$',fontsize = 18)
    fig.savefig("./merge/{}_corner.pdf".format(source))
    
    
    # Retrieve the double data
    with open("./double/{}_bounds_2comp_v3.pickle".format(source), 'r') as pkl_file:
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
    #pemcee_max    = [3.67321752, 1.35584954, 18.0147834, -9.9650754, 4.39663557, 2.58709877, 15.85634216, -11.00979968]
    print pemcee_max

    model_Jup = range(1, 12)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    minorLocator_x = MultipleLocator(1)
    minorLocator_y = MultipleLocator(0.5)
    ax.errorbar(Jup, flux.value, eflux.value, fmt='o', label='$\mathrm{data}$', color='#000000', capsize=0)
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

    ax.plot(plot_Jup, f_inter_pemcee_max(plot_Jup), label=r'$\mathrm{{MCMC}_{double}}$', color='#A3008F')
    ax.plot(plot_Jup, ysingle, label=r'$\mathrm{MCMC_{single}}$', color='#FFA833')
    ax.plot(plot_Jup, f_inter_pemcee_max_w(plot_Jup), linestyle='--', color='#FF5430')
    ax.plot(plot_Jup, f_inter_pemcee_max_c(plot_Jup), linestyle='-.', color='#E96AB2')
    ax.set_xlabel(r'$J_\mathrm{up}$',fontsize=14)
    ax.set_ylabel(r'$I_\mathrm{CO}\;[\mathrm{Jy\;km\;s^{-1}}]$',fontsize=14)
    ax.legend(loc=0, prop={'size':11}, numpoints=1)
    ax.xaxis.set_minor_locator(minorLocator_x)
    ax.yaxis.set_minor_locator(minorLocator_y)
    fig.suptitle('$\mathrm{'+source+'}$',fontsize = 15)
    fig.savefig("./merge/{}_SLED_2comp.pdf".format(source))

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
                                quantiles=(0.16, 0.84), truths=new_pemcee_max_c,
                                truth_color="#E96AB2", color="#198189", bins=24)
    fig.savefig("./merge/{}_corner_2comp_1.pdf".format(source))
    
    fig = corner.corner(chain_warm,
                        labels=[r'$\mathrm{log}_{10}(n_\mathrm{H_2}\;[\mathrm{cm}^{-3}])$',
                                r'$\mathrm{log}_{10}(T_\mathrm{kin}\;[\mathrm{K}])$',
                                r'$\mathrm{log}_{10}({N_\mathrm{CO}}/{\mathrm{d}v}\;[\frac{\mathrm{cm}^{-2}}{\mathrm{km\,s}^{-1}}])$'],
                                show_titles=True, title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 15}, 
                                plot_datapoints=False, range=plot_range, max_n_ticks=6, smooth=0.8,
                                quantiles=(0.16, 0.84), truths=new_pemcee_max_w,
                                truth_color="#FF5430", color="#b1d623", bins=24)
    fig.savefig("./merge/{}_corner_2comp_2.pdf".format(source))

    
    # Print the MCMC results
    chain_plot_cold = np.hstack((chain_cold[:,[0,1,2]], chain_cold[:,[0]]+chain_cold[:,[1]]))
    chain_plot_warm = np.hstack((chain_warm[:,[0,1,2]], chain_warm[:,[0]]+chain_warm[:,[1]]))
    n_c, T_c, N_c, P_c = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                         zip(*np.percentile(chain_plot_cold, [16, 50, 84], axis=0)))
    n_w, T_w, N_w, P_w = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                         zip(*np.percentile(chain_plot_warm, [16, 50, 84], axis=0)))
    
    print "#### cold component - median ####"
    print ' ', n_c[0],' ', T_c[0],' ', N_c[0],' ', P_c[0]
    print '+', n_c[1],'+', T_c[1],'+', N_c[1],'+', P_c[1]
    print '-', n_c[2],'-', T_c[2],'-', N_c[2],'-', P_c[2]
    print "4max", '\n', pemcee_max_c
    print "================================="
    print "#### warm component - median ####"
    print ' ', n_w[0],' ', T_w[0],' ', N_w[0],' ', P_w[0]
    print '+', n_w[1],'+', T_w[1],'+', N_w[1],'+', P_w[1]
    print '-', n_w[2],'-', T_w[2],'-', N_w[2],'-', P_w[2]
    print "4max", '\n', pemcee_max_w
    print "================================="



# See :
# - https://arxiv.org/pdf/1602.01095.pdf
# - https://arxiv.org/pdf/1401.2998.pdf
# - https://arxiv.org/pdf/1212.5955.pdf
# - https://arxiv.org/abs/0809.2337
# plt.figure()
# plt.plot(Jup, model[:8]/model[0])
