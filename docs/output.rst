.. _output:

Output files
============

gsf produces plots for the MCMC covariance matrices, star formation histories, and SED in default. 
gsf also produces the following files, so users can make their own plots for these quantities;

- summary_<ID>.fits
- chain_<ID>_corner.cpkl
- SFH_<ID>.fits
- gsf_spec_<ID>.asdf

See `this notebook <https://github.com/mtakahiro/gsf/blob/version1.4/example/Plot%20SFH%20and%20SED.ipynb>`__ 
for how to use these files.

`this notebook <../example/Plot%20SFH%20and%20SED.ipynb>`__



1.summary file
--------------
summarizes percentiles (16/50/84th) for fitting parameters (e.g., amplitude) 
as well as primary physical parameters (e.g.,stellar mass). Its header also contains meta infos
such as calculation time. 



2.chain file
------------
contains MCMC-chains.


3.SFH file
----------
contains star formation rates (logSFR), stellar masses (logMstel), and metallicity (logZ) 
as a function of lookback time (time) that can be used for plot.


4.Spectral file
---------------
contains the best fit model spectra for each age/tau component, as well as observed fluxes and 
error.
