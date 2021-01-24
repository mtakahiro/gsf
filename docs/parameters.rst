

Parameters
==========

If one needs a new config file

.. code-block:: bash

    python get_configfile.py


Paramter Description
--------------------


**Parameters for the fitting step**

.. list-table::
   :widths: 10 5 20
   :header-rows: 1   
   :class: tight-table   

   * - Parameter
     - Type
     - Description
   * - NMC
     - int
     - No. of iterations for the primary MCMC step.
   * - NWALK
     - int
     - No. of walkers for the primary MCMC step.
   * - NMCZ
     - int
     - No. of iterations for the pre-redshift fitting step.
   * - NWALKZ
     - int
     - No. of walkers for the pre-redshift fitting step.
   * - FNELD
     - int
     - Minimization method in lmfit. For initial values. 0=Powell, 1=Nelder (faster).|
   * - ZVIS
     - int 
     - Visual inspection of redshift. 1=yes, 0=no. If you are not confident with input values (ZGAL, CZ0, CZ1), then one should set this to 1 for iterative fit.
   * - 
     - 
     - 

**Parameters for input data**

.. list-table::
   :widths: 10 5 20
   :header-rows: 0 
   :class: tight-table   

   * - Parameter
     - Type
     - Description
   * - DIR_TEMP
     - str
     - Directory for spectral templates to be stored. If not exist, gsf will create one.
   * - CAT_BB
     - str
     - Broadband photometry catalog. Read '' for its format.
   * - DIR_FILT
     - str
     - Directory for filter response curve files.
   * - FILTER
     - str
     - Filters of broadband photometry. Comma-separated string, where each string should match \*.fil files in DIR_FILT.
   * - DIR_EXTR
     - str
     - Directory for spectroscopic data. If none, gsf will ignore and fit only to broadband data.
   * - SPEC_FILE
     - str 
     - Comma separated list for spectral files. Path is relative to DIR_EXTR.
   * - MORP
     - str
     - Profile shape, if grism spectra are provided. Will be used to convolve templates. "moffat", "gauss", or none.
   * - MORP_FILE
     - str
     - Ascii file for morphology parameters. 
   * - VDISP
     - float
     - Velocity dispersion in km/s. Will be used to convolve templates if MORP=none.


**Parameters for spectral templates**

.. list-table::
   :widths: 10 5 20
   :header-rows: 1   
   :class: tight-table   

   * - Parameter
     - Type
     - Description
   * - AGE
     - str 
     - Set of age pixels, lookback time, in Gyr. Comma-separated.
   * - TAU0
     - int
     - Length for star formation of each age pixel, in Gyr (0.01 to 20Gyr). If 99, CSP is applied. If negative, SSP is applied.
   * - NIMF
     - int 
     - Choice of IMF. 0=Salpeter, 1=Chabrier, 2=Kroupa, 3=van Dokkum, 4=Dave, 5=tabulated, specified in imf.dat.
   * - ADD_LINES
     - int
     - If 1, emission lines will be added to spectral templates. 0=no, 1=yes. Only supported for fsps.
   * - LOGU
     - float
     - Ionizing parameter U, in log, only effective when ADD_LINES==1.
   * - ZMAX
     - float
     - Maximum value for metalicity, in logZ.
   * - ZMIN
     - float
     - Minimum value for metalicity, in logZ.
   * - DELZ
     - float
     - Resolution for metallicity, in logZ.
   * - AVMAX
     - float 
     - Maximum value for Av (dust attenuation in V-band), in mag.
   * - AVMIN
     - float
     - Minimum value for Av (dust attenuation in V-band), in mag.
   * - ZMC
     - int 
     - If 1, set redshift as a free parameter in the primary MCMC step. 0=no, 1=yes.


**Parameters for a specific target**

.. list-table::
   :widths: 10 5 20
   :header-rows: 1   
   :class: tight-table   

   * - Parameter
     - Type
     - Description
   * - ID
     - str
     - Target ID. You can also specify this with an additional argment ("--id").
   * - ZGAL
     - float
     - Initial guess of source redshift. You can skip this if "redshift" column is included in CAT_BB.
   * - CZ0
     - float
     - Initial guess of spectral normalization for G102.
   * - CZ1
     - float
     - Initial guess of spectral normalization for G141.
   * - 
     - 
     - 
   * - 
     - 
     - 

.. list-table::
   :widths: 10 5 20
   :header-rows: 1   
   :class: tight-table   

   * - 
     - 
     - 
   * - 
     - 
     - 

.. list-table::
   :widths: 10 5 20
   :header-rows: 1   
   :class: tight-table   

   * - 
     - 
     - 
   * - 
     - 
     - 

.. list-table::
   :widths: 10 5 20
   :header-rows: 1   
   :class: tight-table   

   * - 
     - 
     - 
   * - 
     - 
     - 
