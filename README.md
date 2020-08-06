# The project of fitting SLED with `PyRADEX`+`emcee`


## Why use `RADEX`+`emcee` rather than using `RADEX` grids?

- This code combines `emcee` and `RADEX`, performing MCMC sampling in the RADEX-parameters space WITHOUT predefined grids, which leads to faster a convergence time and a better sampling of the parameter space.
- This code directly samples the parameter space with the Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler. This allows a much better sampling of the parameter space. 	

## Installation & usage: 


### Dependence:

- `PyRadex`: https://github.com/keflavich/pyradex
- `emcee`: https://github.com/dfm/emcee
- `corner`: https://github.com/dfm/corner.py
- other common packages: `numpy`, `scipy`, `astropy`, `astroquery` 

### How to produce resulting plots:
- The results are stored in the `.pickle` files
- Just run `replot(“source_name”)’ after executing the python script in `IPython`.
- You can always check the `.pickle` files without the need to rerun the whole code again. 

### Change number of walkers and iterations:
- In `emcee_radex.py`, change the numbers between line 404-406.
- In `emcee_radex_2comp.py`, change the numbers between line 540-543.

### Change the solver of the radiation transfer

To use [`myradex`](https://github.com/fjdu/myRadex) (which may be more stable as mentioned in [Ginsberg et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A%26A...586A..50G/abstract)) instead of RADEX, please uncomment the following line:

```
import pyradex.fjdu
```

You may also need to change `pyradex.Radex` to `pyradex.fjdu.Fjdu` , and remove `validate_colliders=False`.


## Directory structure:

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
	- `flux.dat`: **this is the flux data file read by** `emcee_radex.py`;
	- `flux_for2p.dat`: flux data used in `emcee_radex_2comp.py`;

## Fitting example:
![](./fig/NA144.png)


## Citation
Please cite our paper if you find this code useful. The paper also includes CO SLED data for 16 high-redshift submillimeter galaxies: [C. Yang, A. Omont, A. Beelen et al. 2017, A&A, 608, A144](http://adsabs.harvard.edu/abs/2017A%26A...608A.144Y). 

Bibtex:

```
@ARTICLE{2017A&A...608A.144Y,
       author = {{Yang}, C. and {Omont}, A. and {Beelen}, A. and {Gao}, Y. and
         {van der Werf}, P. and {Gavazzi}, R. and {Zhang}, Z. -Y. and
         {Ivison}, R. and {Lehnert}, M. and {Liu}, D. and {Oteo}, I. and
         {Gonz{\'a}lez-Alfonso}, E. and {Dannerbauer}, H. and {Cox}, P. and
         {Krips}, M. and {Neri}, R. and {Riechers}, D. and {Baker}, A.~J. and
         {Micha{\l}owski}, M.~J. and {Cooray}, A. and {Smail}, I.},
        title = "{Molecular gas in the Herschel-selected strongly lensed submillimeter galaxies at z   2-4 as probed by multi-J CO lines}",
      journal = {\aap},
     keywords = {galaxies: high-redshift, galaxies: ISM, infrared: galaxies, submillimeter: galaxies, radio lines: ISM, ISM: molecules, Astrophysics - Astrophysics of Galaxies},
         year = 2017,
        month = dec,
       volume = {608},
          eid = {A144},
        pages = {A144},
          doi = {10.1051/0004-6361/201731391},
archivePrefix = {arXiv},
       eprint = {1709.04740},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2017A&A...608A.144Y},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Authors of the code

Alexandre Beelen (ORCID: 0000-0003-3201-0185); Chentao Yang (ORCID: 0000-0002-8117-9991).
