# The project of fitting SLED with PyRADEX+emcee
 
### Note: 

This code combines `emcee` and `RADEX`, performing MCMC sampling on the RADEX-parameters space WITHOUT predefined grids, which leads to faster a convergence time and a better sampling of the parameter space.

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
Data are published in C. Yang, A. Omont, A. Beelen et al. 2017, A&A in press. 

Cite here: http://adsabs.harvard.edu/abs/2017arXiv170904740Y

### Authors of the code

Alexandre Beelen (ORCID: 0000-0003-3201-0185); Chentao Yang (ORCID: 0000-0002-8117-9991).