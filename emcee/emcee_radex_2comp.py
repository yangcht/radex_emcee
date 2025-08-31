#!/usr/bin/env python
# To run it on the cluster, login and use
#
# > srun -N 1 -c 16 --exclusive emcee_radex_2comp.py
#
# or
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
import logging
import multiprocessing as mp
import os
import warnings
import _pickle as pickle

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import ascii
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize

import emcee
import corner
import pyradex

# ---- Matplotlib: set backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # headless backend for scripts
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# Adopt a standard science plotting style
try:
    import scienceplots  # optional dependency
    plt.style.use(['science', 'no-latex'])
except Exception:
    plt.style.use('seaborn-colorblind')

# Recommend to install the AVHerschey fonts
matplotlib.rcParams.update({
    'font.family': 'AVHershey Complex',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# ------------------------------- Logging & Warnings -------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("emcee_radex_2comp")

# Silence very chatty libraries at INFO/DEBUG level
for noisy in ("matplotlib", "matplotlib.font_manager", "matplotlib.backends",
              "astropy", "pyradex", "corner"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# In recent Matplotlib this also helps:
try:
    matplotlib.set_loglevel("warning")
except Exception:
    pass

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'AVHershey Complex:style=normal:weight=medium'
matplotlib.rcParams['mathtext.it'] = 'AVHershey Complex:style=italic:weight=medium'
matplotlib.rcParams['mathtext.bf'] = 'AVHershey Complex:style=normal:weight=bold'
matplotlib.use('Agg')

# Corner sometimes emits this when sample windows are too tight — safe to ignore
warnings.filterwarnings("ignore", message=".*Too few points to create valid contours.*")

# ------------------------------- Constants & Units -------------------------------
cosmo = FlatLambdaCDM(H0=67.8 * (u.km / u.s / u.Mpc), Om0=0.308)

kms = u.km / u.s
Jykms = u.Jy * kms

# CPU count for multiprocessing
try:
    mp.set_start_method("fork", force=False)  # prefer 'fork' where available
except (RuntimeError, ValueError):
    pass
ncpu = mp.cpu_count()

# Assuming ortho-to-para ratio equals 3
opr = 3.0
fortho = opr / (1.0 + opr)

# ------------------------------- RADEX handle (module-level) -------------------------------
# Each process creates its own Radex instance for portability across start methods.
R = None

def init_radex(tbg=2.7315):
    """Initialise global pyradex.Radex instance."""
    global R
    if R is None:
        R = pyradex.Radex(
            species='co',
            datapath="radex_moldata",
            density={'oH2': fortho * 10.**10.0, 'pH2': (1 - fortho) * 10.**10.0},
            column=10.**6.0,
            temperature=20.0,
            tbackground=tbg,
            deltav=1.0,
            escapeProbGeom='lvg'
        )

# ------------------------------- Models -------------------------------
def model_lvg(Jup, p, R=None):
    # component 1  +  component 2
    log_density_1, log_temperature_1, log_column_1, log_size_1, \
    log_density_2, log_temperature_2, log_column_2, log_size_2 = p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1st component
        R.set_params(density={'oH2': fortho * 10.**log_density_1,
                              'pH2': (1 - fortho) * 10.**log_density_1},
                     column=10.**log_column_1,
                     temperature=10.**log_temperature_1)
        R.run_radex(validate_colliders=False, reuse_last=True, reload_molfile=False)
        result_1 = R.source_line_surfbrightness  # astropy quantity

        # 2nd component
        R.set_params(density={'oH2': fortho * 10.**log_density_2,
                              'pH2': (1 - fortho) * 10.**log_density_2},
                     column=10.**log_column_2,
                     temperature=10.**log_temperature_2)
        R.run_radex(validate_colliders=False, reuse_last=True, reload_molfile=False)
        result_2 = R.source_line_surfbrightness

    Jup_idx = np.asarray(np.int_(Jup)) - 1
    intensity = (result_1[Jup_idx] * (10.**log_size_1 * u.sr) * (1. * kms)).to(Jykms) + \
                (result_2[Jup_idx] * (10.**log_size_2 * u.sr) * (1. * kms)).to(Jykms)
    return intensity.value


def model_single_lvg(Jup, p, R=None):
    log_density, log_temperature, log_column, log_size = p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R.set_params(density={'oH2': fortho * 10.**log_density,
                              'pH2': (1 - fortho) * 10.**log_density},
                     column=10.**log_column,
                     temperature=10.**log_temperature)
        R.run_radex(validate_colliders=False, reuse_last=True, reload_molfile=False)
        result = R.source_line_surfbrightness
    Jup_idx = np.asarray(np.int_(Jup)) - 1
    intensity = (result[Jup_idx] * (10.**log_size * u.sr) * (1. * kms)).to(Jykms)
    return intensity.value
# ------------------------------- Likelihood & Priors -------------------------------
def residual(p, R=None, Jup=None, flux=None, eflux=None):
    model_flux = model_lvg(Jup, p, R)
    return (flux - model_flux) / eflux


def lnlike(p, Jup, flux, eflux, R=None, sigma_floor=1e-12):
    try:
        model_flux = model_lvg(Jup, p, R)
    except ValueError:
        return -np.inf

    # Force float64 and check finiteness
    flux       = np.asarray(flux, dtype=np.float64)
    model_flux = np.asarray(model_flux, dtype=np.float64)
    eflux      = np.asarray(eflux, dtype=np.float64)

    if not (np.all(np.isfinite(flux)) and np.all(np.isfinite(model_flux))):
        return -np.inf

    # Guard against zero/too-small uncertainties
    e = np.maximum(np.abs(eflux), sigma_floor)
    if not np.all(np.isfinite(e)):
        return -np.inf

    # Work with standardized residuals; bail out before squaring if they’re enormous
    r = (flux - model_flux) / e
    max_safe = np.sqrt(np.finfo(np.float64).max) / 10.0  # ~1e153 as a generous cap
    if np.any(~np.isfinite(r)) or np.any(np.abs(r) > max_safe):
        return -np.inf

    # Sum of squares without creating huge intermediates
    chi2 = np.dot(r, r)
    return -0.5 * (chi2 + 2.0 * np.sum(np.log(e)))


def lnprior(p, bounds, T_d=None, R=None):
    # p: [n1, T1, N1, size1, n2, T2, N2, size2] in log10 for n, T, N, size
    log_density_1, log_temperature_1, log_column_1, log_size_1, \
    log_density_2, log_temperature_2, log_column_2, log_size_2 = p

    # Boundaries
    if (np.any(p > bounds[:, 1]) or np.any(p < bounds[:, 0])):
        return -np.inf

    # Force the 2nd component to be warmer
    if log_temperature_2 <= log_temperature_1:
        return -np.inf

    # 9 < log10(N_CO/dv) - log10(n_H2) < 18  (both components)
    if ((log_column_1 - log_density_1) >= 18.0 or (log_column_1 - log_density_1) <= 9.0 or
        (log_column_2 - log_density_2) >= 18.0 or (log_column_2 - log_density_2) <= 9.0):
        return -np.inf

    # cold component size >= warm component size (in log-space: larger means less negative)
    if log_size_1 < log_size_2:
        return -np.inf

    # Priors: Gaussian on T_cold (around T_d) + flat within bounds otherwise
    logp = 0.0
    for idx, (value, bound) in enumerate(zip(p, bounds)):
        if idx == 1 and T_d is not None:  # log10(T_cold)
            T_kin = 10.0 ** value
            if T_d <= 0:
                return -np.inf
            sigma = 1.0 * T_d
            logp += (-0.5 * ((T_kin - T_d) / sigma) ** 2.0
                     - np.log(sigma * np.sqrt(2.0 * np.pi)))
        else:
            # Uniform within bounds (constant; can be omitted since it's a constant offset)
            logp += - (bound[1] - bound[0])
    return logp


def lnprob(p, Jup, flux, eflux, bounds=None, T_d=None):
    lp = lnprior(p, bounds, T_d=T_d, R=R)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnlike(p, Jup, flux, eflux, R=R)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# ------------------------------- I/O helpers -------------------------------
def read_data(filename):
    """Read data file into a pandas-like transposed table."""
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
    """Retrieve redshift and CO data for a particular source."""
    z = data[source]['z']
    line_width = data[source]['line_width']  # not used directly here
    T_d = data[source]['T_d']

    # Select only CO lines
    keys = [key for key in data[source].keys() if 'CO' in key and 'eCO' not in key]
    CO_data = Table(np.asarray([(Jlow + 1, data[source][key], data[source]['e' + key])
                                for Jlow, key in enumerate(keys)
                                if np.isfinite(data[source][key])]),
                    names=['Jup', 'flux', 'eflux'],
                    dtype=[int, float, float])

    Jup = CO_data['Jup'].data
    flux = CO_data['flux'].data * Jykms
    eflux = CO_data['eflux'].data * Jykms
    return z, T_d, line_width, Jup, flux, eflux

def nearest_sample_to_vector(samples, target, metric='mahalanobis', eps=1e-9):
    """
    Return (nearest_sample, index, distance^2) to `target` from `samples`.
    metrics: 'mahalanobis' | 'z' (z-scored Euclidean) | 'euclidean'
    """
    X = np.asarray(samples, dtype=float)
    t = np.asarray(target,  dtype=float)

    if metric == 'mahalanobis':
        C = np.cov(X, rowvar=False)
        # regularize for numerical stability
        C.flat[::C.shape[0] + 1] += eps
        L = np.linalg.cholesky(C)
        delta = (X - t).T                               # shape (D, N)
        z = np.linalg.solve(L, delta)                   # L z = delta
        dist2 = np.sum(z * z, axis=0)                   # length-N
    elif metric == 'z':
        s = np.std(X, axis=0, ddof=1)
        s = np.where(s > 0, s, eps)
        dist2 = np.sum(((X - t) / s) ** 2.0, axis=1)
    else:  # 'euclidean'
        dist2 = np.sum((X - t) ** 2.0, axis=1)

    i = int(np.argmin(dist2))
    return X[i], i, float(dist2[i])

# ------------------------------- Plotting helpers -------------------------------
def replot(source, representative='median', metric='mahalanobis'):
    """Replot SLED and corner plots from saved pickle.

    representative: 'median' (default) -> nearest posterior sample to marginal medians (via `metric`)
                    'map'              -> maximum a posteriori sample from chain (argmax log-prob)
    metric: 'mahalanobis' | 'z' | 'euclidean'  (used only when representative='median')
    """
    with open(f"./double/{source}_bounds_2comp.pickle", 'rb') as pkl_file:
        (source, z, bounds, T_d,
         (Jup, flux, eflux), (popt, pcov), pmin, theta_med, (chain, lnprobability)) = pickle.load(pkl_file)

    init_radex(tbg=2.7315 * (1 + z))
    R.set_params(tbg=2.7315 * (1 + z))

    # Build flatchain & within-1σ slice
    flatchain = chain.reshape((-1, 8))
    lnp = lnprobability.reshape((-1, 1))
    lower, upper = np.percentile(flatchain, [16, 84], axis=0)
    mask = np.all((flatchain > lower) & (flatchain < upper), axis=1)
    narrow_flatchain = flatchain[mask]
    narrow_lnp = lnp[mask]

    # Existing MAP-ish point from your code (keep as-is)
    pemcee_max = narrow_flatchain[narrow_lnp.argmax()] if len(narrow_flatchain) else theta_med
    pemcee_max_c = pemcee_max[:4]
    pemcee_max_w = pemcee_max[4:]

    # Nearest-to-median sample (Mahalanobis by default)
    theta_star, idx_star, d2_star = nearest_sample_to_vector(
        flatchain, theta_med, metric=metric
    )
    theta_star_c = theta_star[:4]
    theta_star_w = theta_star[4:]

    # ---- Choose which representative to plot (NEW) ----
    rep = str(representative).lower()
    if rep in ('map', 'max', 'maximum_likelihood'):
        theta_ref   = pemcee_max
        theta_ref_c = pemcee_max_c
        theta_ref_w = pemcee_max_w
        label_main, label_warm, label_cold = r'$\mathrm{MCMC\text{-}Max}$', r'$\mathrm{Max-warm}$', r'$\mathrm{Max-cold}$'
        color_main, color_warm, color_cold = '#FFA833', '#fcc82d', '#ff7b33'
    else:  # default: 'median'
        theta_ref   = theta_star
        theta_ref_c = theta_star_c
        theta_ref_w = theta_star_w
        label_main, label_warm, label_cold = r'$\mathrm{MCMC\text{-}nearest\ Median}$', r'$\mathrm{Med-warm}$', r'$\mathrm{Med-cold}$'
        color_main, color_warm, color_cold = '#FFA833', '#fcc82d', '#ff7b33'

    # ---------------- SLED plot (plot ONLY the chosen representative) ----------------
    model_Jup = np.arange(1, 12)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    minorLocator_x = MultipleLocator(1)
    minorLocator_y = MultipleLocator(0.5)
    ax.errorbar(Jup, flux.value, eflux.value, fmt='o', ms=2, color='#000000', capsize=0, label=r'$\mathrm{data}$', zorder=15)
    ax.xaxis.set_minor_locator(minorLocator_x)
    ax.yaxis.set_minor_locator(minorLocator_y)
    plot_Jup = np.arange(model_Jup.min(), model_Jup.max(), 0.05)

    # Interpolators for chosen representative
    f_inter_ref_total = interp1d(model_Jup, model_lvg(model_Jup, theta_ref, R), kind='cubic')
    f_inter_ref_c     = interp1d(model_Jup, model_single_lvg(model_Jup, theta_ref_c, R), kind='cubic')
    f_inter_ref_w     = interp1d(model_Jup, model_single_lvg(model_Jup, theta_ref_w, R), kind='cubic')

    # Plot selected representative only
    ax.plot(plot_Jup, f_inter_ref_total(plot_Jup), label=label_main, linewidth=1.5, color=color_main, zorder=15)
    ax.plot(plot_Jup, f_inter_ref_w(plot_Jup),     label=label_warm, linestyle='--', color=color_warm, zorder=15)
    ax.plot(plot_Jup, f_inter_ref_c(plot_Jup),     label=label_cold, linestyle='-.', color=color_cold, zorder=15)

    # Keep your random posterior draws overlay (underneath)
    if len(narrow_flatchain) > 0:
        inds = np.random.randint(len(narrow_flatchain), size=min(200, len(narrow_flatchain)))
        for ind in inds:
            sample = narrow_flatchain[ind]
            model_flux = model_lvg(model_Jup, sample, R)
            f_inter = interp1d(model_Jup, model_flux, kind='cubic')
            ax.plot(plot_Jup, f_inter(plot_Jup), color='#f5ec42', alpha=0.1, zorder=1)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel(r'$J_\mathrm{up}$',fontsize=14)
    ax.set_ylabel(r'$I_\mathrm{CO}\;[\mathrm{Jy\;km\;s^{-1}}]$',fontsize=14)
    ax.legend(loc='best', prop={'size':8}, numpoints=1)
    fig.suptitle(r'$\mathrm{'+source+'}$', fontsize=16)
    fig.savefig(f"./double/{source}_SLED_2comp.pdf", bbox_inches='tight')
    plt.close(fig)

    # ---------------- Full 8D corner (truths = chosen representative) ----------------
    chain_plot = np.hstack((flatchain[:, [0, 1, 2, 3]], flatchain[:, [4, 5, 6, 7]]))
    labels8 = [
        r'$\mathrm{log}_{10}(n_\mathrm{H_2,\,c}\,[\mathrm{cm}^{-3}])$',
        r'$\mathrm{log}_{10}(T_\mathrm{kin,\,c}\,[\mathrm{K}])$',
        r'$\mathrm{log}_{10}(N_\mathrm{CO,\,c}/\mathrm{d}v\,[\mathrm{cm}^{-2}/\mathrm{km\,s}^{-1}])$',
        r'$\mathrm{log}_{10}(\mathrm{[size,\,c\,sr^{-1}]})$',
        r'$\mathrm{log}_{10}(n_\mathrm{H_2,\,w}\,[\mathrm{cm}^{-3}])$',
        r'$\mathrm{log}_{10}(T_\mathrm{kin,\,w}\,[\mathrm{K}])$',
        r'$\mathrm{log}_{10}(N_\mathrm{CO,\,w}/\mathrm{d}v\,[\mathrm{cm}^{-2}/\mathrm{km\,s}^{-1}])$',
        r'$\mathrm{log}_{10}(\mathrm{[size,\,w\,sr^{-1}]})$',
    ]
    plot_range8 = [(1.9, 7.1), (1, 3.02), (14.5, 19.5), (-12.5, -8.5),
                   (1.9, 7.1), (1, 3.0), (14.5, 19.5), (-12.5, -8.5)]

    truths_for_corner = np.hstack((theta_ref_c, theta_ref_w))
    fig = corner.corner(
        chain_plot,
        labels=labels8,
        show_titles=True, title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 15},
        plot_datapoints=False, range=plot_range8, max_n_ticks=6, smooth=0.8,
        quantiles=[0.15865, 0.50, 0.84135], truths=truths_for_corner,
        truth_color=color_main, color="#2B61DD", bins=24
    )
    fig.suptitle(r'$\mathrm{' + source + '}$', fontsize=16)
    fig.savefig(f"./double/{source}_corner_2comp_all.pdf", bbox_inches='tight')
    plt.close(fig)

    # ---------------- Publication 3D corners (truths = chosen representative) ----------------
    chain_cold = flatchain[:, [0, 1, 2]]
    chain_warm = flatchain[:, [4, 5, 6]]
    new_pemcee_max_c = theta_ref_c[:3]   # keep variable names, but now consistent with chosen rep
    new_pemcee_max_w = theta_ref_w[:3]
    plot_range3 = [(1.9, 7.1), (1, 3.02), (14.5, 19.5)]

    fig = corner.corner(
        chain_cold,
        labels=[r'$\mathrm{log}_{10}(n_\mathrm{H_2}\,[\mathrm{cm}^{-3}])$',
                r'$\mathrm{log}_{10}(T_\mathrm{kin}\,[\mathrm{K}])$',
                r'$\mathrm{log}_{10}(N_\mathrm{CO}/\mathrm{d}v\,[\mathrm{cm}^{-2}/\mathrm{km\,s}^{-1}])$'],
        show_titles=True, title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 15},
        plot_datapoints=False, range=plot_range3, max_n_ticks=6, smooth=0.8,
        quantiles=[0.15865, 0.50, 0.84135], truths=new_pemcee_max_c,
        truth_color=color_cold, color="#198189", bins=24
    )
    fig.savefig(f"./double/{source}_corner_2comp_1.pdf", bbox_inches='tight')
    plt.close(fig)

    fig = corner.corner(
        chain_warm,
        labels=[r'$\mathrm{log}_{10}(n_\mathrm{H_2}\,[\mathrm{cm}^{-3}])$',
                r'$\mathrm{log}_{10}(T_\mathrm{kin}\,[\mathrm{K}])$',
                r'$\mathrm{log}_{10}(N_\mathrm{CO}/\mathrm{d}v\,[\mathrm{cm}^{-2}/\mathrm{km\,s}^{-1}])$'],
        show_titles=True, title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 15},
        plot_datapoints=False, range=plot_range3, max_n_ticks=6, smooth=0.8,
        quantiles=[0.15865, 0.50, 0.84135], truths=new_pemcee_max_w,
        truth_color=color_warm, color="#b1d623", bins=24
    )
    fig.savefig(f"./double/{source}_corner_2comp_2.pdf", bbox_inches='tight')
    plt.close(fig)

    # ---------------- Print median ± 1σ ----------------
    chain_cold_P = np.hstack((chain_cold, chain_cold[:, [0]] + chain_cold[:, [1]]))
    chain_warm_P = np.hstack((chain_warm, chain_warm[:, [0]] + chain_warm[:, [1]]))  # log P = log n + log T

    n_c, T_c, N_c, P_c = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                             list(zip(*np.percentile(chain_cold_P, [16, 50, 84], axis=0))))
    n_w, T_w, N_w, P_w = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                             list(zip(*np.percentile(chain_warm_P, [16, 50, 84], axis=0))))

    print("#### cold component - median ####")
    print(' ', n_c[0], ' ', T_c[0], ' ', N_c[0], ' ', P_c[0])
    print('+', n_c[1], '+', T_c[1], '+', N_c[1], '+', P_c[1])
    print('-', n_c[2], '-', T_c[2], '-', N_c[2], '-', P_c[2])
    print("4max", '\n', new_pemcee_max_c)
    print("=================================")
    print("#### warm component - median ####")
    print(' ', n_w[0], ' ', T_w[0], ' ', N_w[0], ' ', P_w[0])
    print('+', n_w[1], '+', T_w[1], '+', N_w[1], '+', P_w[1])
    print('-', n_w[2], '-', T_w[2], '-', N_w[2], '-', P_w[2])
    print("4max", '\n', new_pemcee_max_w)
    print("=================================")

# ------------------------------- Main -------------------------------
def main():
    init_radex()
    data = read_data("../data/flux_for2p.dat")

    if not os.path.exists("./double"):
        os.makedirs("./double")

    for source in data.keys():
        logger.info('Processing %s', source)

        # Retrieve data & set CMB background
        z, T_d, line_width, Jup, flux, eflux = get_source(source, data)
        R.set_params(tbg=2.7315 * (1 + z))

        # Physical/numerical bounds (narrowed, literature-guided)
        # n_H2    = 10^1.5 -- 10^7.0 cm^-3
        # T_kin   = T_CMB -- 10^3 K (in log10)
        # N_CO/dv = 10^14.5 -- 10^19.5 cm^-2 (km/s)^-1
        # size (sr) around an angular size scale (broad prior)
        R_angle = ((7 / (cosmo.angular_diameter_distance(z).value * 1000.0)) ** 2 * np.pi) * 10
        # Assume 7 kpc size and μ=10 lensing magnification

        bounds = np.array([
            [1.5, 7.0],                                    # log n_H2 (cold)
            [np.log10(2.7315 * (1 + z)), 3.0],            # log T_kin (cold)
            [14.5, 19.5],                                  # log N_CO/dv (cold)
            [np.log10(R_angle) - 9, np.log10(R_angle) + 9],  # log size (cold)
            [1.5, 7.0],                                    # log n_H2 (warm)
            [np.log10(2.7315 * (1 + z)), 3.0],            # log T_kin (warm)
            [14.5, 19.5],                                  # log N_CO/dv (warm)
            [np.log10(R_angle) - 9, np.log10(R_angle) + 9]   # log size (warm)
        ])

        # Initial guess near literature values
        p0 = [
            1.9,   # log n_H2 cold
            1.2,   # log T_kin cold
            16.4,  # log N_CO/dv cold
            -12.1, # log size cold
            3.9,   # log n_H2 warm
            2.5,   # log T_kin warm
            17.5,  # log N_CO/dv warm
            -12.1  # log size warm
        ]

        # curve_fit to find a reasonable start
        opt_fun = lambda p,  log_density_1, log_temperature_1, log_column_1, log_size_1, \
                             log_density_2, log_temperature_2, log_column_2, log_size_2: \
               model_lvg(p, [log_density_1, log_temperature_1, log_column_1, log_size_1,
                             log_density_2, log_temperature_2, log_column_2, log_size_2], R=R)

        try:
            popt, pcov = curve_fit(
                opt_fun, Jup, flux.value, sigma=eflux.value,
                p0=p0, bounds=list(zip(*bounds))
            )
            logger.info("    curve_fit : %s", popt)
        except Exception as e:
            logger.warning("    curve_fit failed: %s", e)
            popt, pcov = p0, None

        # Minimize negative posterior to sanity-check lnprob
        nll = lambda p, Jup, flux, eflux: -lnprob(p, Jup, flux, eflux, bounds, T_d)
        result = minimize(nll, popt, args=(Jup, flux.value, eflux.value), bounds=bounds)
        pmin = result.x
        logger.info("    minimize : %s", pmin)

        # ------------------ MCMC ------------------
        ndim = len(popt)
        nwalkers = 400
        n_iter_burn = 100
        n_iter_walk = 1000

        # Random starting positions around curve_fit result
        pos = [popt + 1e-3 * np.random.randn(ndim) for _ in range(nwalkers)]

        with mp.Pool(processes=min(ncpu, nwalkers), initializer=init_radex,
                     initargs=(2.7315 * (1 + z),)) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob,
                args=(Jup, flux.value, eflux.value),
                kwargs={'bounds': bounds, 'T_d': T_d},
                pool=pool
            )
            logger.info("    burning samples")
            state = sampler.run_mcmc(pos, n_iter_burn, progress=False)
            sampler.reset()

            logger.info("    walking")
            sampler.run_mcmc(state, n_iter_walk, progress=False)

        # Results
        chain = sampler.get_chain()                  # (steps, walkers, ndim)
        lnprobability = sampler.get_log_prob()       # (steps, walkers)
        flatchain = sampler.get_chain(flat=True)     # (steps*walkers, ndim)
        lnp = sampler.get_log_prob(flat=True)[:, None]

        # Representative point: posterior median
        theta_med = np.percentile(flatchain, 50, axis=0)

        # Save everything needed by replot()
        with open(f"./double/{source}_bounds_2comp.pickle", 'wb') as pkl_file:
            pickle.dump(
                (source, z, bounds, T_d,
                 (Jup, flux, eflux), (popt, pcov), pmin, theta_med, (chain, lnprobability)),
                pkl_file
            )

        # Print compact summary (median ± 1σ) for (log n, log T, log N, log P)
        chain_cold = flatchain[:, [0, 1, 2]]
        chain_warm = flatchain[:, [4, 5, 6]]
        chain_cold_P = np.hstack((chain_cold, chain_cold[:, [0]] + chain_cold[:, [1]]))
        chain_warm_P = np.hstack((chain_warm, chain_warm[:, [0]] + chain_warm[:, [1]]))  # log P = log n + log T

        n_c, T_c, N_c, P_c = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                 list(zip(*np.percentile(chain_cold_P, [16, 50, 84], axis=0))))
        n_w, T_w, N_w, P_w = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                 list(zip(*np.percentile(chain_warm_P, [16, 50, 84], axis=0))))

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("xxx:", source, '\n', "xxx: minimised results")
        new_pmin_c = np.hstack((pmin[:3], pmin[0] + pmin[1]))
        new_pmin_w = np.hstack((pmin[4:7], pmin[4] + pmin[5]))
        print("xxx:", new_pmin_c, '\n', new_pmin_w)

        print("xxx: emcee results")
        print("xxx:", "n_H2");   print("xxx:", n_c, '\n', "xxx:", n_w)
        print("xxx:", "T_kin");  print("xxx:", T_c, '\n', "xxx:", T_w)
        print("xxx:", "N_CO/dv");print("xxx:", N_c, '\n', "xxx:", N_w)
        print("xxx:", "P");      print("xxx:", P_c, '\n', "xxx:", P_w)

        # Optional: quick plot now
        # replot(source)

# ------------------------------- Entry point -------------------------------
if __name__ == '__main__':
    main()

# See :
# - https://arxiv.org/pdf/1602.01095.pdf
# - https://arxiv.org/pdf/1401.2998.pdf
# - https://arxiv.org/pdf/1212.5955.pdf
# - https://arxiv.org/abs/0809.2337
