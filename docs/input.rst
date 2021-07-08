.. _input:

Input files
===========

Broadband catalog
-----------------
Gsf reads the ascii catalog specified with ``CAT_BB`` in the configuration file. 
The catalog format is similar to EAzY (`Brammer et al. 2008 <http://adsabs.harvard.edu/abs/2008ApJ...686.1503B>`__) 
and FAST (`Kriek et al. 2009 <http://adsabs.harvard.edu/abs/2009ApJ...700..221K}>`__) and should be formatted as;

ID redshift [flux for filter 1] [flux error for filter 1]...

For example, if ``FILTER`` in your configuration is assigned "202,205,..." (where the numbers correspond to filter response files; :ref:`Filter`), the header of the broadband catalog has to be set as;

# ID redshift F202 E202 F205 E205...

(See also the example data provided in "example" directory.)

The redshift column in the ascii catalog is optional. If the redshift column is not included, redshift needs to be specified by ``ZGAL`` in the configuration file. 

The flux unit needs to be in f\_nu, with a magnitude zero point :math:`m_0=25`, i.e.

.. math::

    m = -2.5 \log_{10}(f_\nu)+m_0

.. _Filter:

Filter response curve
---------------------

``FILTER`` array correspond to the response curve files stored in ``DIR_FILT``. 
For example, if one of the keywords in ``FILTER`` is "205", 
gsf will look into ``DIR_FILT`` directory to find a filter response file named "205.fil". 
Standard filter response curve files are available in the package (cloned from EAzY), while users can add their own filter response files with the format as below;

# [Column_number] [Wavelength_in_AA] [Response]


Spectral data
-------------
When spectral data are provided, gsf reads the ascii spectral file for the target object, in ``DIR_EXTR`` in the configuration file. The file name should be specified by ``SPEC_FILE``, that is formatted as;

# [Wavelength_in_AA] [Flux_nu] [Error_flux_nu]

Units of flux and error need to be in f_nu with a magnitude zero point :math:`m_0=25`.

For grism spectra, users are asked to provide morphological parameters of the target to appropriately convolve spectral templates to the size of the source. 
In the current version, gsf convolves model templates either with a 1-dimensional Moffat function,

.. math::

    f(r;\alpha,\gamma) =  A \Big[1+\Big({r^2\over{\gamma^2}}\Big)\Big]^{-\alpha}

or Gaussian,

.. math::
    
    f(r;\gamma) =  A \exp{\Big({-r^2\over{2\gamma^2}}\Big)}

The parameters should be stored in the ascii file speficied by ``MORP_FILE``, with the following format;

# A gamma alpha

for both cases (i.e. put an arbitrary number for alpha if Gaussian), where A is a normalization constant.
