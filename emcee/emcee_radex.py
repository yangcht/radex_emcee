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
#SBATCH --mail-user=xxx

# autopep8 --ignore E26 emcee_radex.py
import os
import pandas as pd
import numpy as np
import multiprocessing
import emcee
import pickle
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
import warnings
from astropy.io import ascii
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import logging
import corner

# For the pyradex package (ensure it's installed in your environment)
import pyradex

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Adopt a science-friendly plotting style with consistent fonts
try:
    import scienceplots  # optional dependency
    plt.style.use(['science', 'no-latex'])
except Exception:
    plt.style.use('seaborn-colorblind')

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Make multiprocessing behave reasonably across platforms
try:
    multiprocessing.set_start_method("fork", force=False)
except (RuntimeError, ValueError):
    pass

# Set up logging (quiet 3rd-party debug spam)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("emcee_radex_single")

# Turn down noisy libraries
for noisy in ("matplotlib", "matplotlib.font_manager", "matplotlib.backends",
              "astropy", "pyradex", "corner"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Optional: also lower Matplotlib’s own level (newer versions)
try:
    import matplotlib as _mpl
    _mpl.set_loglevel("warning")
except Exception:
    pass

# Optional: corner sometimes warns when contours are under-sampled
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*Too few points to create valid contours.*")

# Initialize constants and cosmology
kms = u.km / u.s
Jykms = u.Jy * kms
# to match the cosmo in the CO paper
cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Om0=0.308)

opr = 3
fortho = opr / (1 + opr)

# pyradex handle.  Each process needs its own instance when using the
# default 'spawn' start method on some platforms, so we create it via an
# initializer.
R = None


def init_radex(tbg=2.7315):
    """Initialise global pyradex.Radex instance for each process."""
    global R
    if R is None:
        R = pyradex.Radex(
            species='co',
            datapath="radex_moldata",
            density={'oH2': fortho * 10.**10.0, 'pH2': (1 - fortho) * 10.**10.0},
            column=10.0**6.0,
            temperature=20.0,
            tbackground=tbg,
            deltav=1.0,
            escapeProbGeom='lvg'
        )

# Function definitions
def model_lvg(Jup, params, R=None):
    log_density, log_temperature, log_column, log_size = params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R.set_params(density={'oH2': fortho * 10.**log_density, 'pH2': (1 - fortho) * 10.**log_density},
                     column=10.**log_column,
                     temperature=10.**log_temperature)
        R.run_radex(validate_colliders=False, reuse_last=True, reload_molfile=False)
        result = R.source_line_surfbrightness
    intensity = (result[np.asarray(np.int_(Jup)) - 1] * (10.**log_size * u.sr) * (1. * kms)).to(Jykms)
    return intensity.value

def lnlike(p, Jup, flux, eflux, R=None, sigma_floor=1e-12):
    """Likelihood function"""
    try:
        model_flux = model_lvg(Jup, p, R)
    except ValueError:
        return -np.inf

    # Cast to float64 and basic finiteness checks
    flux       = np.asarray(flux, dtype=np.float64)
    model_flux = np.asarray(model_flux, dtype=np.float64)
    eflux      = np.asarray(eflux, dtype=np.float64)

    if not (np.all(np.isfinite(flux)) and np.all(np.isfinite(model_flux))):
        return -np.inf

    # Guard against zero/too-small uncertainties
    e = np.maximum(np.abs(eflux), sigma_floor)
    if not np.all(np.isfinite(e)):
        return -np.inf

    # Standardized residuals
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        r = (flux - model_flux) / e

    # Bail out cleanly if residuals are non-finite or astronomically large
    if not np.all(np.isfinite(r)):
        return -np.inf
    max_safe = np.sqrt(np.finfo(np.float64).max) / 10.0  # generous headroom
    if np.any(np.abs(r) > max_safe):
        return -np.inf

    # chi2 and log term (use 2*log(sigma) instead of log(sigma**2))
    chi2 = np.dot(r, r)
    log_term = 2.0 * np.sum(np.log(e))

    return -0.5 * (chi2 + log_term)

def lnprior(p, bounds, R=None):
    if (np.any(p > bounds[:, 1]) or np.any(p < bounds[:, 0])):
        return -np.inf
    # physical constraint: 10.0 < log(N_CO/dv) - log(n_H2) < 17.5
    if (p[2] - p[0] >= 17.5) or (p[2] - p[0] <= 10.0):
        return -np.inf
    # limit maximum optical depth
    try:
        if max_tau(p, R) > 100:
            return -np.inf
    except ValueError:
        return -np.inf
    return 0.0


def max_tau(p, R=None):
    """Return maximum optical depth for parameter set."""
    log_density, log_temperature, log_column, _ = p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R.set_params(density={'oH2': fortho * 10.**log_density,
                              'pH2': (1 - fortho) * 10.**log_density},
                     column=10.**log_column,
                     temperature=10.**log_temperature)
        R.run_radex(validate_colliders=False, reuse_last=True, reload_molfile=False)
        tau = np.amax(R.tau)
    return tau

def lnprob(p, Jup, flux, eflux, bounds=None):
    lp = lnprior(p, bounds, R=R)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p, Jup, flux, eflux, R=R)

def read_data(filename):
    ##############################################################
    #  This will match the table in ../table/flux.data.          #
    #  Note that you can add as many CO transitions as you can.  #
    #  But you will always need to have                          #
    #     four columns for the flux and errors for               #
    #     the lines of CI(1-0) and CI(2-1).                      #
    ##############################################################
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        if not line.strip().startswith("#") and line.strip():
            data_lines.append(line.strip())

    first_data_line = data_lines[0].split()
    num_columns = len(first_data_line)

    fixed_columns = ["SOURCE", "z", "D_L", "line_width"]
    num_dynamic_columns = (num_columns - 8) // 2

    co_columns = []
    for i in range(num_dynamic_columns):
        co_columns.append(f"CO_J_{i+1}")
        co_columns.append(f"eCO_J_{i+1}")

    all_columns = fixed_columns + co_columns + ["CI_1", "eCI_1", "CI_2", "eCI_2"]

    data = []
    for line in data_lines:
        split_line = line.split()
        data.append(split_line)

    if any(len(row) != len(all_columns) for row in data):
        raise ValueError("Number of columns in data rows does not match the expected number of columns")

    df = pd.DataFrame(data, columns=all_columns)

    numerical_columns = all_columns[1:]
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index('SOURCE', inplace=True)
    return df.T

def get_source(source, data):
    """Retrieve redshift and CO data for a particular source"""
    z = data[source]['z']
    line_width = data[source]['line_width']
    keys = [key for key in data[source].keys() if 'CO' in key and 'eCO' not in key]
    CO_data = Table(np.asarray([(Jlow + 1, data[source][key], data[source]['e' + key])
                                for Jlow, key in enumerate(keys) if np.isfinite(data[source][key])]),
                    names=['Jup', 'flux', 'eflux'], dtype=[int, float, float])
    Jup = CO_data['Jup'].data
    flux = CO_data['flux'].data * Jykms
    eflux = CO_data['eflux'].data * Jykms
    return z, line_width, Jup, flux, eflux

def replot(source):
    plt.ion()
    # Retrieve the data
    with open("./single/{}_bounds.pickle".format(source), 'rb') as pkl_file:
        (source, z, bounds, (Jup, flux, eflux), (popt, pcov), pmin, theta_med, (chain, lnprobability)) = pickle.load(pkl_file)

    init_radex(tbg=2.7315 * (1 + z))

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

    # compute the models for chi^2 (pmin), median (pemcee) and maximum-likelihood (pemcee_max)
    f_inter_pmin = interp1d(model_Jup, model_lvg(model_Jup, pmin, R), kind='cubic')
    f_inter_med  = interp1d(model_Jup, model_lvg(model_Jup, theta_med, R), kind='cubic')
    f_inter_pemcee_max = interp1d(model_Jup, model_lvg(model_Jup, pemcee_max, R), kind='cubic')

    # plot the models onto the CO SLED
    ax.plot(plot_Jup, f_inter_pmin(plot_Jup), label=r'$\mathrm{{\chi}^2}$', linestyle='--', color='#2B61DD')
    #ax.plot(plot_Jup, f_inter_pemcee(plot_Jup), label=r'$\mathrm{median_{MCMC}}$', linestyle='--', color='#2B61DD')
    ax.plot(plot_Jup, f_inter_pemcee_max(plot_Jup), label=r'$\mathrm{MCMC}$', color='#FFA833')

    # plot the 200 "good models" within the [16, 84] quartile
    inds = np.random.randint(len(narrow_flatchain), size=200)
    for ind in inds:
        sample = narrow_flatchain[ind]
        f_inter_pemcee_sample = interp1d(model_Jup, model_lvg(model_Jup, sample, R), kind='cubic')
        ax.plot(plot_Jup, f_inter_pemcee_sample(plot_Jup), color='#f5ec42', alpha=0.1)

    ax.set_xlabel(r'$J_\mathrm{up}$',fontsize=14)
    ax.set_ylabel(r'$I_\mathrm{CO}\;[\mathrm{Jy\;km\;s^{-1}}]$',fontsize=14)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc=0, prop={'size':12}, numpoints=1)
    fig.suptitle(r'$\mathrm{'+source+'}$', fontsize=16)
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
                        quantiles=[0.15865, 0.50, 0.84135], truths=pemcee_max, truth_color="#FFA833", color="#2B61DD", bins=24)
    fig.suptitle(r'$\mathrm{'+source+'}$', fontsize=16)
    fig.savefig("./single/{}_corner_full.pdf".format(source))

    # plots for publication, remove size from the plot
    plot_range=[(1.9,7.1),(1,3.02),(14.5, 19.5)]
    fig = corner.corner(flatchain[:,[0,1,2]],
          labels=[r'$\mathrm{log}_{10}(n_\mathrm{H_2}\;[\mathrm{cm}^{-3}])$',
                  r'$\mathrm{log}_{10}(T_\mathrm{kin}\;[\mathrm{K}])$',
                  r'$\mathrm{log}_{10}({N_\mathrm{CO}}/{\mathrm{d}v}\;[\frac{\mathrm{cm}^{-2}}{\mathrm{km\,s}^{-1}}])$'],
          show_titles=True, title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 15}, 
          plot_datapoints=False, range=plot_range, max_n_ticks=6, smooth=0.6,
          quantiles=[0.15865, 0.5, 0.84135], truths=pemcee_max[:3], truth_color="#FFA833", color="#2B61DD", bins=24)
    fig.suptitle(r'$\mathrm{'+source+'}$', fontsize=16)
    fig.savefig("./single/{}_corner.pdf".format(source))

    # Print the MCMC results
    flatchain_pressure = np.hstack((flatchain[:,[0,1,2]], flatchain[:,[0]]+flatchain[:,[1]]))
    n_c, T_c, N_c, P_c= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                        list(zip(*np.percentile(flatchain_pressure, [16, 50, 84], axis=0))))

    print("median")
    print(' ', n_c[0],' ', T_c[0],' ', N_c[0],' ', P_c[0])
    print('+', n_c[1],'+', T_c[1],'+', N_c[1],'+', P_c[1])
    print('-', n_c[2],'-', T_c[2],'-', N_c[2],'-', P_c[2])
    print("4max", '\n', pemcee_max)

def main():
    init_radex()
    data = read_data("../data/flux.dat")

    if not os.path.exists("./single"):
        os.makedirs("./single")

    for source in data.columns:
        logger.info(f'Processing {source}')
        z, line_width, Jup, flux, eflux = get_source(source, data)

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

        R.set_params(tbg=2.7315 * (1 + z))

        # Assuming, 7 kpc size and mu=10 lensing magnification
        R_angle = ((7 / (cosmo.angular_diameter_distance(z).value * 1000.0)) ** 2 * np.pi) * 10

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
                           [np.log10(R_angle) - 4, np.log10(R_angle) + 4]])

        p0 = [  4.0,    # np.log10(total_density)
                1.4,    # np.log10(temperature)
               17.8,    # 10**16.75*300, # np.log10(column)
              -9.85]  # np.log10(size)

        # Simple curve_fit to find a starting point
        # Using lambda to get the R object through
        p0 = np.clip(p0, bounds[:, 0], bounds[:, 1])

        opt_fun = lambda p, log_density, log_temperature, log_column, log_size: \
            model_lvg(p, [log_density, log_temperature, log_column, log_size], R=R)

        try:
            popt, pcov = curve_fit(opt_fun, Jup, flux.value, sigma=eflux.value, p0=p0, bounds=list(zip(*bounds)))
            logger.info(f"curve_fit : {popt}")
        except RuntimeError:
            logger.warning("curve_fit : failed")
            popt = p0
            pcov = None

        p = popt
        nll = lambda p, Jup, flux, eflux: -lnprob(p, Jup, flux, eflux, bounds)
        result = minimize(nll, p, args=(Jup, flux.value, eflux.value), bounds=bounds)
        pmin = result.x
        logger.info(f"minimize : {pmin}")

        ndim = len(popt)
        #################### Define the number of walkers here
        nwalkers = 100       # 100 walkers
        n_iter_burn = 100    # burning phase, number of iterations = 100
        n_iter_walk = 500    # walking phase, number of iterations = 500

        # Random starting positions
        pos = [popt + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]

        # Multithread
        with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), nwalkers),
                                  initializer=init_radex,
                                  initargs=(2.7315 * (1 + z),)) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob,
                args=(Jup, flux.value, eflux.value),
                kwargs={'bounds': bounds},
                pool=pool
            )
            logger.info("burning samples")
            state = sampler.run_mcmc(pos, n_iter_burn, progress=False)
            sampler.reset()

            logger.info("walking")
            sampler.run_mcmc(state, n_iter_walk, progress=False)

        # emcee v3: pull results this way
        chain = sampler.get_chain()               # (steps, walkers, ndim)
        lnprobability = sampler.get_log_prob()    # (steps, walkers)
        flatchain = sampler.get_chain(flat=True)  # (steps*walkers, ndim)

        # Representative point to save with results (posterior median)
        theta_med = np.percentile(flatchain, 50, axis=0)

        with open(f"./single/{source}_bounds.pickle", 'wb') as pkl_file:
            pickle.dump(
                (source, z, bounds,
                 (Jup, flux, eflux), (popt, pcov), pmin, theta_med, (chain, lnprobability)),
                pkl_file
            )

        chain_plot = np.hstack((flatchain[:, [0, 1, 2]], flatchain[:, [0]] + flatchain[:, [1]]))
        new_pmin = np.hstack((pmin[:3], pmin[0] + pmin[1]))

        # Quick plot the model
        # replot(source)

        n_h2, T_kin, N_co, Pres = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                      list(zip(*np.percentile(chain_plot, [16, 50, 84], axis=0))))

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print(f"xxx: {source}\nxxx: minimised results")
        print(f"xxx: {new_pmin}")
        print("xxx: emcee results")
        print("xxx: n_H2")
        print(f"xxx: {n_h2}")
        print("xxx: T_kin")
        print(f"xxx: {T_kin}")
        print("xxx: N_CO/dv")
        print(f"xxx: {N_co}")
        print("xxx: P")
        print(f"xxx: {Pres}")

if __name__ == '__main__':
    plt.ion()
    main()
