#!/usr/bin/env python
# To run it on the cluster, login and use
#
# > srun -N 1 -c 16 --exclusive emcee_radex.py
#
# to launch it on a node with 16 core, or use
#
# > sbatch emcee_radex.py
#
#
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --partition=cpu_only
#SBATCH --account=cyang
#SBATCH --exclusive
#SBATCH --mail-type=END
#SBATCH --mail-user=chentao.yang@ias.u-psud.fr

# autopep8 --ignore E26 emcee_radex.py
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


def model_lvg_tau(p, R=None):
    log_density, log_temperature, log_column, log_size = p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R.set_params(density={'oH2':fortho*10.**log_density,'pH2':(1-fortho)*10.**log_density},
                     column=10.**log_column,
                     temperature=10.**log_temperature)
        R.run_radex(validate_colliders=False,
                    reuse_last=True, reload_molfile=False)
        result = R.tau # Do not use get_table()
    return np.amax(result)


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


def lnprior(p, bounds, R=None):
    # 1. Check the boundaries
    """Uniform prior"""
    if (np.any(p > bounds[:, 1])
            or np.any(p < bounds[:, 0])
            or np.any(p[2]-p[0] >= 17.5)     #see bounds from dv/dr
            or np.any(p[2]-p[0] <= 10.0)):
        logp = -np.inf
    else:
        logp = -np.sum(np.log(bounds.dot([-1, 1]))) #a flat prior
    # 2. add bounds for tau, should be < 100
    #   try:
    #       tau_max = model_lvg_tau(p, R)
    #   except ValueError:
    #       return -np.inf
    #   
    #   if (tau_max > 100):
    #       return -np.inf
    return logp


def lnprob(p, Jup, flux, eflux, bounds=None):
    lp = lnprior(p, bounds, R=R)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p, Jup, flux, eflux, R=R)


def read_data(filename):
    """Read data into a comprehensible panda frame"""

    ascii_data = ascii.read(
        filename, names=[
            "SOURCE", "z", "D_L", "line_width",
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

    return z, line_width, Jup, flux, eflux


# the pyradex.Radex object MUST be declared at the module level for
# lnprob to work properly
#  ....  Setup the Radex parameters

# Note that N_CO is totaly degenerated with deltav, so in
# principle we should fit for N_CO/deltav
R = pyradex.Radex(species='co', datapath="radex_moldata",
                  density={'oH2':fortho*10.**10.0,'pH2':(1-fortho)*10.**10.0},
                  column=10.0**6.0,
                  temperature=20.0,
                  tbackground=2.7315,
                  deltav=1.0,
                  escapeProbGeom='lvg')


def replot(source):

    plt.ion()
    # Retrieve the data
    with open("./single/{}_bounds.pickle".format(source), 'r') as pkl_file:
        (source, z, bounds, (Jup, flux, eflux), (popt, pcov), pmin, pemcee, (chain, lnprobability)) =  pickle.load(pkl_file)

    R.set_params(tbg=2.7315 * (1 + z))
    
    # Get the max posterior within +/-1 sigma range
    flatchain = chain.reshape((chain.shape[0]*chain.shape[1]),4) 
    lnp = lnprobability.reshape((lnprobability.shape[0]*lnprobability.shape[1]),1)
    lower, upper     = np.percentile(flatchain, [16, 84],axis=0)
    narrow_flatchain = flatchain[(flatchain[:,0] > lower[0]*1) & (flatchain[:,0] < upper[0]*1) & \
                                 (flatchain[:,1] > lower[1]*1) & (flatchain[:,1] < upper[1]*1) & \
                                 (flatchain[:,2] > lower[2]*1) & (flatchain[:,2] < upper[2]*1) & \
                                 (flatchain[:,3] > lower[3]*1) & (flatchain[:,3] < upper[3]*1) ]
    narrow_lnp       =       lnp[(flatchain[:,0] > lower[0]*1) & (flatchain[:,0] < upper[0]*1) & \
                                 (flatchain[:,1] > lower[1]*1) & (flatchain[:,1] < upper[1]*1) & \
                                 (flatchain[:,2] > lower[2]*1) & (flatchain[:,2] < upper[2]*1) & \
                                 (flatchain[:,3] > lower[3]*1) & (flatchain[:,3] < upper[3]*1) ]   
    pemcee_max       = narrow_flatchain[narrow_lnp.argmax()]
    
    model_Jup = range(1, 12)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    minorLocator_x = MultipleLocator(1)
    minorLocator_y = MultipleLocator(0.5)
    ax.errorbar(Jup, flux.value, eflux.value, fmt='o', color='#000000', capsize=0, label=r'$\mathrm{data}$')
    ax.xaxis.set_minor_locator(minorLocator_x)
    ax.yaxis.set_minor_locator(minorLocator_y)
    plot_Jup = np.arange(min(model_Jup), max(model_Jup), 0.05) # smoothing the model line
    f_inter_pmin = interp1d(model_Jup, model_lvg(model_Jup, pmin, R), kind='cubic')
    f_inter_pemcee = interp1d(model_Jup, model_lvg(model_Jup, pemcee, R), kind='cubic')
    f_inter_pemcee_max = interp1d(model_Jup, model_lvg(model_Jup, pemcee_max, R), kind='cubic')
    #ax.plot(plot_Jup, f_inter_pemcee(plot_Jup), label=r'$\mathrm{median_{MCMC}}$', linestyle='--', color='#2B61DD')
    ax.plot(plot_Jup, f_inter_pemcee_max(plot_Jup), label=r'$\mathrm{MCMC}$', color='#FFA833')
    ax.set_xlabel(r'$J_\mathrm{up}$',fontsize=14)
    ax.set_ylabel(r'$I_\mathrm{CO}\;[\mathrm{Jy\;km\;s^{-1}}]$',fontsize=14)
    ax.legend(loc=0, prop={'size':12}, numpoints=1)
    fig.suptitle('$\mathrm{'+source+'}$')
    fig.savefig("./single/{}_SLED.pdf".format(source))
    
    # plots for the full corner
    plot_range=[(1.9,7.1),(1,3.02),(14.5, 19.5),(-12.5,-8.5)]
    fig = corner.corner(flatchain,
                        labels=[r'$\mathrm{log}_{10}(n_\mathrm{H_2}\;[\mathrm{cm}^{-3}])$',
                                r'$\mathrm{log}_{10}(T_\mathrm{kin}\;[\mathrm{K}])$',
                                r'$\mathrm{log}_{10}({N_\mathrm{CO}}/{\mathrm{d}v}\;[\frac{\mathrm{cm}^{-2}}{\mathrm{km\,s}^{-1}}])$',
                                r'$\mathrm{log}_{10}(\mathrm{[size\,sr^{-1}]})$'],
                        show_titles=True, title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 15}, 
                        plot_datapoints=False, range=plot_range, max_n_ticks=6, smooth=0.6,
                        quantiles=(0.15865, 0.84135), truths=pemcee_max, truth_color="#FFA833", color="#2B61DD", bins=24)
    fig.suptitle('$\mathrm{'+source+'}$',fontsize = 16)
    fig.savefig("./single/{}_corner_full.pdf".format(source))

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
    fig.savefig("./single/{}_corner.pdf".format(source))

    # Print the MCMC results
    flatchain_pressure = np.hstack((flatchain[:,[0,1,2]], flatchain[:,[0]]+flatchain[:,[1]]))
    n_c, T_c, N_c, P_c= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                        zip(*np.percentile(flatchain_pressure, [16, 50, 84], axis=0)))
    
    print "median"
    print ' ', n_c[0],' ', T_c[0],' ', N_c[0],' ', P_c[0]
    print '+', n_c[1],'+', T_c[1],'+', N_c[1],'+', P_c[1]
    print '-', n_c[2],'-', T_c[2],'-', N_c[2],'-', P_c[2]
    print "4max", '\n', pemcee_max


def main():

    data = read_data("../data/flux.dat")
    
    if not os.path.exists("./single"):
        os.makedirs("./single")

    for source in data.keys():
    # if True:
    #     source = 'NCv1.143'

        logger.info('Processing {}'.format(source))

        delta_v = 1 * kms  # do not care actually, fully degenerate
                           # with the column density

        # TODO: would it help to use the CI data ?
        # If 2 lines probably yes, otherwise, must be degenerate with
        # the abondance

        # Retrieve data
        z, line_width, Jup, flux, eflux = get_source(source, data)

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
                           [-20, -5]])  # log size
                           
        R_angle = ((7/(cosmo.angular_diameter_distance(z).value*1000.0))**2 * np.pi)*10
        # Assuming, 7 kpc size and mu=10 lensing magnification
        
        # basic starting point
        p0 = [5.6,     # np.log10(total_density)
              24,      # temperature
              18,      # 10**16.75*300, # np.log10(column)
              np.log10(R_angle), ]  # np.log10(size)

        # As suggested by https://arxiv.org/pdf/1212.5955.pdf
        # Further narrowing down according to Zhang et al, A&A 568, A122 (2014)
        # remove unrealistic ranges
        # Ranges:
        # n_H2    = 10^2.0 -- 10^7.0 cm^-3
        # T_kin   = T_CMB -- 1000 K
        # N_CO/dv = 10^15.5 -- 10^19.5 cm^-2 (km/s)^-1
        # dv/dr   = 0.1 -- 1000 (Tunnard+2016, Xco=5e-5), saying r ~ 1-5 kpc, d_V = 250-700 km/s
        #  --> 6.2e13 < N_CO/n_H2 < 6.2e17 
        #  --> due to lensing uncertainties, add +/-50 lensing uncertainty factor, multiply d_V = d_V = 250-700 km/s
        #  --> 10.0 < log10(N_CO/dv) - log10(n_H2) < 17.5
        # Additional constrains:
        # N_CO/(n_H2*Xco) < 2 R = 10 kpc (assuming disk is =<5 kpc, Xco=5e-5)
        #  -->   N_CO/n_H2 < 5e19, within the dv/dr range.
        
        bounds = np.array([[2.0, 7.0],
                           [np.log10(2.7315 * (1 + z)), 3.0],
                           [15.5, 19.5],
                           [np.log10(R_angle)-4, np.log10(R_angle)+4]])

        p0 = [  4.0,    # np.log10(total_density)
                1.4,    # np.log10(temperature)
               17.8,    # 10**16.75*300, # np.log10(column)
              -9.85, ]  # np.log10(size)

        # Simple curve_fit to find a starting point
        # Using lambda to get the R object through
        opt_fun = lambda p, log_density, log_temperature, log_column, log_size: \
            model_lvg(p, [log_density, log_temperature, log_column, 
                          log_size], R=R)

        try:
            popt, pcov = curve_fit(opt_fun, Jup, flux.value,
                                   sigma=eflux.value, p0=p0,
                                   bounds=zip(*bounds))
            logger.info("    curve_fit : {}".format(popt))
        except RuntimeError:
            logger.warn("    curve_fit : failed")
            popt = p0
            pcov = None

        # minimize do not work well far from the solution, so use the
        # curve_fit solution to test the lnprob function...
        p = popt
        nll = lambda p, Jup, flux, eflux, : - \
            lnprob(p, Jup, flux, eflux, bounds)
        result = minimize(nll, p,
                          args=(Jup, flux.value, eflux.value),
                          bounds=bounds)
        pmin = result.x
        logger.info("    minimize : {}".format(pmin))

        # Do the heavy computation
        ndim = len(popt)
        nwalkers = 400  # 400 walkers
        # Random starting positions
        pos = [popt + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=(Jup, flux.value, eflux.value),
                                        kwargs={'bounds': bounds},
                                        threads=ncpu)
        # Burning time
        logger.info("    burning samples")
        pos, prob, state = sampler.run_mcmc(
            pos, 100, storechain=False)  # now 100, will be 1000
        sampler.reset()
        # Sampling time
        logger.info("    walking")
        result = sampler.run_mcmc(pos, 1000)  # now 1000, will be 5000
        pemcee = np.percentile(sampler.flatchain, [50], axis=0)[0]

        chain, lnprobability = sampler.chain, sampler.lnprobability

        with open("./single/{}_bounds.pickle".format(source), 'w') as pkl_file:
            pickle.dump((source, z, bounds,
                         (Jup, flux, eflux), (popt, pcov), pmin, pemcee, (chain, lnprobability)),
                        pkl_file)

        # Quick plot the model
        # replot(source)
        
        # Print only the best fitted range of n_H2, T_kin, N_CO and Pressure
        chain_plot = np.hstack((sampler.flatchain[:,[0,1,2]], sampler.flatchain[:,[0]]+sampler.flatchain[:,[1]]))
        new_pmin = np.hstack((pmin[:3],pmin[0]+pmin[1]))  
        
        n_h2, T_kin, N_co, Pres = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                      zip(*np.percentile(chain_plot, [16, 50, 84], axis=0)))
        
        # Output the best fit
        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        print "xxx:", source, '\n' "xxx: minimised results"
        print "xxx:", new_pmin
        # Print the MCMC results for +/- 1 sigma range of the paremeters
        print "xxx: emcee results"
        print "xxx:", "n_H2"
        print "xxx:", n_h2
        print "xxx:", "T_kin"    
        print "xxx:", T_kin
        print "xxx:", "N_CO/dv"     
        print "xxx:", N_co
        print "xxx:", "P"        
        print "xxx:", Pres

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
