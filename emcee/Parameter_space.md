# SLED fitting with PyRADEX+emcee
#### Parameters:
- `n_H2`:  10^2.0 -- 10^7.0 cm^-3
- `T_kin`:  T_CMB -- 10^3.0 (1000) K
- `N_CO/dv`:  10^15.5 - 10^19.5 cm^-2/(km/s) 
- `size`:  log10(R_angle) +/- 4 considering uncertainties of size and magnification

#### Constraints:
- `dv/dr`: 0.1 -- 1000 km/s pc^-1 
	- `10 < log10(N_CO/dv) - log10(n_H2) < 17.5`
- `size limit`
- Two components:
	`T_d ~ T_k` for cold component, using a Gaussian prior.
	- `v0` : in prior Tcold < Twarm 
	- `v1` : in the model, twarm = tcold+deltaT, deltaT>0 
	- `v2` : in the model, twarm = tcold+deltaT, nh2(warm) = nH2(cold) + deltanH2, deltaT>0, deltanH2>0
	- **`v3` : in prior Tcold < Twarm; SizeCold>SizeWarm**
	- `v4` : in prior Tcold < Twarm; SizeCold>SizeWarm; Pcold < Pwarm

