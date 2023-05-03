.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>

gsf 
~~~~~~~~~~~~~~~~~~~~~~
version 1.7 and after

- SED fitting code used in `Morishita et al. (2018) <http://adsabs.harvard.edu/abs/2018ApJ...856L...4M>`__ and `Morishita et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019ApJ...877..141M/abstract>`__. 
- The main purpose is to explore galaxy properties with flexible-form star formation histories.
- This code utilizes fsps and BPASS templates; fsps templates are generated within the code by using python-fsps.
- This code was originally designed for HST grism data; now it can be used for a geranal purpose (e.g., broadband photometry, JWST prism).


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
