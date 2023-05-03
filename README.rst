.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>

gsf 
~~~~~~~~~~~~~~~~~~~~~~
version 1.7 and after

- SED fitting code used in `Morishita et al. (2018) <http://adsabs.harvard.edu/abs/2018ApJ...856L...4M>`__ and `Morishita et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019ApJ...877..141M/abstract>`__. 
- This code utilizes fsps or BPASS templates; fsps templates are generated within the code by using python-fsps.
- This code was originally designed for HST grism data; now it can be used for a geranal purpose (e.g., broadband photometry, JWST prism).


Why do I need gsf?
~~~~~~~~~~~~~~~~~~

The strength of gsf lies in its flexibility to model the galaxy spectrum by adopting a flexible approach to the galaxy's star formation history 
and metallicity enrichment history. This flexibility is a huge advantage and allows the user to explore a wide range of galaxy properties, 
including nebular emission and redshift (although the latter is not recommended due to the significant CPU cost).

However, this flexibility comes at the expense of increased parameters, which could lead to a situation where the final results are 
significantly unconstrained. Therefore, it is advised that users have a sufficient number of data points spread across wide wavelength range, 
rest-UV-to-NIR, or high-S/N spectrum where continuum features are visible.

For users without a sufficient number of data points, gsf can still offer SED fit at a reduced number of parameters, for example, 
fixed ranges of age, dust attenuation, metallicity, etc. Therefore, users are recommended to understand the advantage of gsf 
over other SED fitting tools. For example, if the user is interested in quickly getting reasonable inference on stellar masses of many galaxies 
from a survey, FAST/FAST++ (Kriek+09) offers excellent affordability. On the other hand, gsf can offer the user to explore a wider parameter range, 
and can allow the user to *maximize* their uncertainty estimates, of galaxy properties that could have been potentially underestimated with the simplified 
configuration assumed in general SED fitting tools.


========================================================================================


Demonstration
~~~~~~~~~~~~~
.. image:: ./sample.png

- Fitting movie can be found `here <https://youtu.be/pdkA9Judd-M>`__.

Pre-requirement
~~~~~~~~~~~~~~~

- astropy
- Pandas
- multiprocess
- lmfit (no older than v1.0.0)
- emcee (no older than v3.0.0)
- corner
- python-fsps (v0.3.0)
- fsps


Installation & Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required packages will be installed by;

.. code-block:: bash

    git clone https://github.com/mtakahiro/gsf.git
    cd gsf 
    pip install -r requirements.txt 

If a user prefers to install in a new conda environment, execute the following before the command above;

.. code-block:: bash

    conda create -n gsf python=3.10
    conda activate gsf

Then, make sure to install `fsps <https://github.com/cconroy20/fsps>`__ and `python-fsps <https://github.com/dfm/python-fsps>`__ by following their instruction.

Lastly, check your installation by;

.. code-block:: bash

    python -m pytest

Done!! :tada:

Examples
~~~~~~~~
An example data set is stored at at `gsf_examples <https://github.com/mtakahiro/gsf_examples/tree/master/example/>`__

.. code-block:: bash

    python run_gsf.py test.input <flag>


If one needs a new config file

.. code-block:: bash

    python example/get_configfile.py

Take a look at `notebooks <https://github.com/mtakahiro/gsf_examples/tree/master/example/>`__ for other use cases.


Execution flag
~~~~~~~~~~~~~~
- 0: Generating z=0 templates (takes a while if MILES is specified in the fsps config file). Start from here if any critical parameter (e.g., Z-range, age bins) in config file is changed. This will then automatically proceed to the next step.
- 1: Redshift template to z=z_input, by using existing z=0 templates (from Step 0). This will then automatically proceed to the next step.
- 2: Fitting part, by using existing redshifted templates (from step1). This will then automatically proceed to the next step.
- 3: Creates SFH and SED plots by using the fitting results.


Release notes
~~~~~~~~~~~~~
- V1.8: JWST prism spectrum can be fit. Usability in Notebook has been improved. Logger has been implemented (thanks to Antonio Addis).
- V1.7.4: pytest has been implemented.
- V1.6: Emission lines can be added to the stellar templates. This is controlled by two parameter, Aneb (amplitude) and logU.
- Far-IR data set can be fit simultaneously with a simple gray body spectrum (to be published in a future version).
- V1.3: log-space samplings for amplitude parameters are implemented.
- V1.2: BPASS templates can also be implemented. Those who wish to try the functionality, please contact the author.
- Data set without grism data, despite the code's name, can be also provided.


Citation
~~~~~~~~~

.. code-block:: bash
    
    @ARTICLE{2019ApJ...877..141M,
        author = {{Morishita}, T. and {Abramson}, L.~E. and {Treu}, T. and {Brammer}, G.~B. and {Jones}, T. and {Kelly}, P. and {Stiavelli}, M. and {Trenti}, M. and {Vulcani}, B. and {Wang}, X.},
            title = "{Massive Dead Galaxies at z {\ensuremath{\sim}} 2 with HST Grism Spectroscopy. I. Star Formation Histories and Metallicity Enrichment}",
        journal = {\apj},
        keywords = {galaxies: abundances, galaxies: evolution, galaxies: formation, galaxies: high-redshift, galaxies: star formation, Astrophysics - Astrophysics of Galaxies},
            year = 2019,
            month = jun,
        volume = {877},
        number = {2},
            eid = {141},
            pages = {141},
            doi = {10.3847/1538-4357/ab1d53},
    archivePrefix = {arXiv},
        eprint = {1812.06980},
    primaryClass = {astro-ph.GA},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2019ApJ...877..141M},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
