
Paramter Description
--------------------


**Parameters for the fitting step:**

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
     - Minimization method in lmfit. For initial values. 0=Powell, 1=Nelder (faster).
   * - ZVIS
     - int 
     - Visual inspection of redshift. 1=yes, 0=no. If you are not confident with input values (ZGAL, CZ0, CZ1), then this should be set to 1 for iterative fit.
   * - 
     - 
     - 

**Parameters for input data:**

.. list-table::
   :widths: 10 5 20
   :header-rows: 1
   :class: tight-table   

   * - Parameter
     - Type
     - Description
   * - DIR_OUT
     - str
     - Directory for output products. If not exist, gsf will create one.
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
   * - SKIPFILT
     - str
     - List of filters that are skipped in the fitting (e.g., for IRAC excess). Comma-separated string.
   * - SNLIM
     - int
     - SN limit for data points. If the SN is below this, then eflux of the data point is used as an upper limit.
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
     * - 
     - 
     - 


**Parameters for spectral templates:**

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
   * - AGEFIX
     - str 
     - (Optional) Subset of age pixels that are used in fitting. Lookback time, in Gyr. Comma-separated.
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
   * - ZFIX
     - float
     - (Optional) Metallicity will be fixed to this value if specified. In logZ.
   * - ZEVOL
     - int
     - (Optional) If 1, metallicity for each age pixel will be set as a free parameter. If not, metallicity will be universal to all age pixels (or fixed if ZFIX is provided).
   * - AMAX
     - str 
     - (Optional) Maximum value for amplitude, in normal logarithmic scale.
   * - AMIN
     - str 
     - (Optional) Minimum value for amplitude, in normal logarithmic scale.
   * - AVMAX
     - float 
     - (Optional) Maximum value for Av (dust attenuation in V-band), in mag.
   * - AVMIN
     - float
     - (Optional) Minimum value for Av (dust attenuation in V-band), in mag.
   * - ZMC
     - int 
     - If 1, redshift is set as a free parameter in the primary MCMC step. Otherwise, redshift is fixed to the input parameter.
   * - F_MDYN
     - int 
     - If 1, gsf uses dynamical mass (M_dyn column in CAT_BB) as a prior. Currently not supported.
   * - BPASS
     - int 
     - If 1, BPASS templates will be used. Currently not supported.
     * - 
     - 
     - 


**Parameters for a specific target:**

.. list-table::
   :widths: 10 5 20
   :header-rows: 1   
   :class: tight-table   

   * - Parameter
     - Type
     - Description
   * - ID
     - str
     - Target ID. You can also specify this by adding "--id" argment (i.e. python run_gsf.py <config_file> <Executing-flag> --id <ID>).
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


**Parameters for functional-form SFHs:**

.. list-table::
  :widths: 10 5 20
  :header-rows: 1   
  :class: tight-table   

  * - Parameter
    - Type
    - Description
  * - SFH_FORM
    - int
    - 1: Tau-model, 4: Delayed tau-model, 5: Delayed tau-model with a transition, based on fsps.
  * - NPEAK
    - int
    - Number of components for SFHs combined. (e.g., if 2, then two of SFH_FORM will be combined.)
  * - AGEMAX
    - float
    - Maximum age for the functional form SFH, in log Gyr.
  * - AGEMIN
    - float
    - Minimum age for the functional form SFH, in log Gyr.
  * - DELAGE
    - float
    - Delta age for the age parameter, in log Gyr.
  * - TAUMAX
    - float
    - Maximum tau for the functional form SFH, in log Gyr.
  * - TAUMIN
    - float
    - Minimum tau for the functional form SFH, in log Gyr.
  * - DELTAU
    - float
    - Delta age for the tau parameter, in log Gyr.
  * - 
    - 
    - 


**Parameters for far-infrared components:**
(Beta implimentation from version1.4)

.. list-table::
   :widths: 10 5 20
   :header-rows: 1   
   :class: tight-table   

   * - Parameter
     - Type
     - Description
   * - FIR_FILTER
     - str 
     - Filters of FIR photometry. Comma-separated string, where each string should match \*.fil files in DIR_FILT.
   * - CAT_BB_DUST
     - str 
     - Directory for the FIT photometric catalog, in the same format as for CAT_BB.
   * - TDUST_HIG
     - float
     - Maximum temperature.
   * - TDUST_LOW
     - float
     - Minimum temperature.
   * - TDUST_DEL
     - float
     - Delta T for temperature paramter, in Kelvin.
   * - DIR_DUST
     - str
     - Directory for FIR templates.
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
