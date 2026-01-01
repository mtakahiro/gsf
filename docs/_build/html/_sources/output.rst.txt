.. _output:

Output files
============

Gsf generates plots of the MCMC covariance matrices, star formation histories, and SED in default. 
It also generates the following files, so you can make your own plots of these quantities;

- summary_<ID>.fits
- chain_<ID>_corner.cpkl
- SFH_<ID>.fits
- gsf_spec_<ID>.asdf

See `this notebook <https://github.com/mtakahiro/gsf/blob/master/example/Plot%20SFH%20and%20SED.ipynb>`__ 
for how to use these files.


1.summary file
--------------
stores statistical values (16/50/84th) for fitting parameters (e.g., amplitude) 
as well as primary physical parameters (e.g.,stellar mass). The header also contains meta info
such as calculation time. 


2.chain file
------------
stores MCMC-chains.


3.SFH file
----------
stores star formation rates (logSFR), stellar masses (logMstel), and metallicity (logZ) 
as a function of lookback time (time). Star formation rate is the average SFR in the last `TSET_SFR` (default: 0.1Gyr) of the posetrior SFH.


4.Spectral file
---------------
stores the best fit model spectra for each age/tau component, as well as observed fluxes and 
error.

- Absolute UV magnitude, Muv, is calculated for each realization in the range of rest-frame wavelength range of 1250 - 1650AA using the full posterior SED spectrum. 
- UV beta slope is calculated for each realization by fitting a single slope over a rest-frame wavelength range of 1650 - 2300AA.