.. _input:

Input files
===========


Broadband catalog
-----------------
gsf reads an ascii catalog specified in CAT\_BB in the configuration file. The catalog format is similar to EAzY and FAST \citep{brammer08,kriek09}, and should be;

# ID redshift [flux for filter 1] [flux error for filter 1]...

while redshift column is optional.

The flux unit has to be in f\_nu, with a magnitude zero point :math:`m_0=25`, i.e.

.. math::

    m = -2.5 \log_{10}(f_\nu)+m_0

FILTER array must correspond to response curve files in DIR\_FILT. For example, if one of FILTER keywords is "205", then gsf will look into DIR\_FILT directory to find a filter response file "205.fil", whose format should be in;

# Column_number Wavelength_in_AA Response 

Standard filter response curve files are contained in the package (cloned from EAzY), while users can add their own filter files in the format explained above.
gsf will find the column with ID that matches "ID" in the configuration file. 


Spectral data
-------------
gsf reads an ascii spectral file of for the target object, in DIR\_EXTR in a configuration file. The file should be specified in [SPEC\_FILE], whose formats are;

#  Wavelength_in_AA Flux_nu Error_in_flux 

The unit of flux and error has to be in f_nu with a magnitude zero point :math:`m_0=25`.

For grism spectra, users are asked to provide morphological parameters of the target. In the current version, gsf convolves model templates either with a 1-dimensional Moffat function,

.. math::

    f(r;\alpha,\gamma) =  A \Big[1+\Big({r^2\over{\gamma^2}}\Big)\Big]^{-\alpha}

or Gaussian,

.. math::
    
    f(r;\gamma) =  A \exp{\Big({-r^2\over{2\gamma^2}}\Big)}

The parameters should be stored in an ascii file, [MORP\_FILE], in the following format;

# A gamma alpha

for both cases (i.e. put a random number for alpha if gaussian), where A is a normalization constant.
