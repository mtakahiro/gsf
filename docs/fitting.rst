.. _fitting:

Fitting process
===============

gsf fits synthetic data points (generated from model templates) to input observed data points 
based on the minimization of the following equation (as in posterior_flexible.py);

.. math::
    residual_i = (f_\mathrm{model,i} - f_\mathrm{obs,i})^2 / \sigma_\mathrm{obs,i}^2

where :math:`i` runs over :math:`n` data points, and :math:`f` as flux and :math:`\sigma` as 1-sigma error.

Then, log likelihood is calculated as;

.. math::
    lnlike =  -0.5 \left[ \sum_{i}^{n} \left( residual_i^2 + ln (2 \pi \sigma_i^2) \right) - 2 \chi_\mathrm{log nd} \right]

where :math:`\chi_\mathrm{log nd}` is contributiong from non-detection data points (:math:`f/\sigma<SN_\mathrm{limit}`);

.. math::
    \chi_\mathrm{log nd} = \sum_{i}^{n} ln \left( \sqrt{ \frac{\pi}{2}} \sigma_\mathrm{obs,i} 
    \left(1 + \mathrm{erf} (\frac{\sigma_i SN_\mathrm{limit} - f_\mathrm{model,i}}{\sqrt{2}\sigma_i}) \right) \right)

where erf is the error function. (See Appendix in `Sawicki 2012 <https://ui.adsabs.harvard.edu/abs/2012PASP..124.1208S/abstract>`__ for the mathematical proof for the non-detection part.)


Then log posterior is calculated by;

.. math::
    lnpost = lnlike + lnprior

where lnprior is log prior (see :doc:`prior`).