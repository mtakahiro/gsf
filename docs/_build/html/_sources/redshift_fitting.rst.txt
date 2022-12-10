.. _redshift_fitting:

Redshift Fitting
================
While gsf is now able to search the redshift grid during the fiting process, 
it was not originally designed to efficiently estimate the posterior for redshift. 
Users can turn this redshift fitting functionality by turning ``F_ZMC`` to 1, 
with ``ZMCMIN`` and ``ZMCMAX`` properly specified too, in the input configuration file.

To minimize the disk space usage, gsf generates templates only at ``ZMC`` specified in the input file. 
How it works with redshift search is, during the fitting process, when gsf detects a large shift 
(specifically, larger than the limiting value specified by ``deltaz_lim`` in `function_class`) in redshift from ``ZMC``,
gsf calls the filter convolution function (`function.filconv`) to recalculate the model broadband flux for the given model, 
to better reflect the current place of each walker in the parameter space. ``ZMC`` is also replaced to this new redshift at 
this point. When the shift is small, gsf interpolates the model fluxes for the given set of filters 
after applying the shift to wavelength. 

This extra step allows more accurate parameter search for gsf with redshift as a variable, but significantly slows down the computing speed, by a factor up to 10. 
This overhead is primarily caused by the number of data points in the model templates and filter curves. 
To minimize the computation time, users may want to consder using sparse templates, by using ``--delwave`` argument (delta wave, in Aungstrome) when they run run_gsf.py 
script, or by directly passing the same parameter to `maketemp` function.

