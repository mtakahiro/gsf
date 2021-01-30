.. _prior:

Prior
=====
gsf calculates posterior probability through an MCMC process, 
whose concept is based on the Bayes theorem.

For each parameter set in the minimizer, prior is set (posterior_flexible.py).
In the current version, prior for most of the parameters (amplitude, metallicity, 
dust attenuation, redshift) are set flat, so that;

.. math::
    prior(\theta) &= 1\ \mathrm{for}\ \theta_\mathrm{min} < \theta < \theta_\mathrm{max} \\
                  &= 0\ \mathrm{for\ else}


