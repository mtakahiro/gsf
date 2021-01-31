.. _prior:

Prior
=====
gsf calculates posterior probability through an MCMC process, 
whose concept is based on the Bayes theorem.

For each parameter set in the minimizer, prior is set (posterior_flexible.py).
In the current version, prior for most of the parameters (amplitude, metallicity, 
dust attenuation, redshift) are set flat, so that;

.. math::
    prior(\theta) &= 1\ \mathrm{if}\ \theta_\mathrm{min} < \theta < \theta_\mathrm{max} \\
                  &= 0\ \mathrm{else}.


Below is the list of parameter ranges set in gsf.

**Parameter ranges:**

.. list-table::
   :widths: 10 5 20
   :header-rows: 1   
   :class: tight-table   

   * - Parameter
     - Range
     - Description
   * - A
     - [AMIN:AMAX]
     - Amplitude, in logarithmic space. AMIN and AMAX are set to -3 and 3 in default. 
   * - Z
     - [ZMIN:ZMAX]
     - Metallicity, in logarithmic space. 
   * - AV
     - [AVMIN:AVMAX]
     - Dust attenuation in V-band, in linear space. AVMIN and AVMAX are set to 0 and 4 in default. 
   * - z
     - [ZGAL-EZL:ZGAL+EZH]
     - Redshift, in linear space. EZL and EZH are set to 0.3. Can be specified in the configuration file.
   * - AGE (tau-model)
     - [AGEMIN:AGEMAX]
     - Age of tau model, in logarithmic space. Range can be specified by AGEMIN and AGEMAX.
   * - TAU (tau-model)
     - [TAUMIN:TAUMAX]
     - Age of tau model, in logarithmic space. Range can be specified by TAUMIN and TAUMAX.
