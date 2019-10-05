# The project of fitting SLED with PyRADEX+emcee
 
### Note: 

- Note that the code was using Python 2, some modifications are needed if running under Python3, for example, in `emcee_radex.py`:
	- line 25, use `import _pickle as pickle` in Python3
	- line 200, use `with open("./single/{}_bounds.pickle".format(source), 'rb') as pkl_file:`
	- line 416, use `with open("./single/{}_bounds.pickle".format(source), 'wb') as pkl_file:`
 
- This code combines `emcee` and `RADEX`, performing MCMC sampling on the RADEX-parameters space WITHOUT predefined grids, which leads to faster a convergence time and a better sampling of the parameter space.
- Simply use `replot` function to generate the figures.

### Dependence:

- PyRadex: https://github.com/keflavich/pyradex
- emcee: https://github.com/dfm/emcee
- corner: https://github.com/dfm/corner.py

### Directory structure:

- `README.md`: this file;
- `emcee`
	- `radex_moldata`: the molecular data;
	- `results`: the pickle files storing the MCMC results;
		- `single`: results of one-component fittings;
		- `double`: results of two-component fittings;
	- `emcee_radex.py`: one-component fitting code;
	- `emcee_radex_2comp.py`: warm + cold components, **in prior Tcold < Twarm; SizeCold>SizeWarm**;
- `data`
	- `flux.note`: explaining the data in Yang+2017;
	- `flux.dat`: flux data used in `emcee_radex.py`;
	- `flux_for2p.dat`: flux data used in `emcee_radex_2comp.py`;

### Citation
Data are published in C. Yang, A. Omont, A. Beelen et al. 2017, A&A, 608, A144.

***Please cite our paper (http://adsabs.harvard.edu/abs/2017A%26A...608A.144Y) if you find this code useful.***

### Authors of the code

Alexandre Beelen (ORCID: 0000-0003-3201-0185); Chentao Yang (ORCID: 0000-0002-8117-9991).
