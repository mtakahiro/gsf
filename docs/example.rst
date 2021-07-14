.. _example:


Examples
========

On-the-fly run
--------------

Example data are stored in ./gsf/examples/

.. code-block:: bash

    python run_gsf.py test.input <Execution-flag>


If you need a new config file (\*.input), execute

.. code-block:: bash

    python get_configfile.py


Execution flag
~~~~~~~~~~~~~~
- 0: Generating templates at z=0 (takes a while if MILES). Start from here if parameter in config file is changed. Then go to 1.
- 1: Redshift template to z=ZGAL, and prepare mock photometry that matches to the input filters and spectra, using pre-existing z=0 templates (from step0). Then go to 2.
- 2: Fitting part, using pre-existing z=z_input templates (from Step1). If ZVIS==1, gsf will ask you if the initial redshift fit is reasonable. Then go to 3.
- 3: Only plot SFH and SED using existing result files.
- 6: Plot physical parameters and SED (optional).


Appendicies
-----------

A. Specify target id
~~~~~~~~~~~~~~~~~~~~

You can speficy the target id from the command line. This way, you would not need to make a bunch of config files for each target.

.. code-block:: bash

    python run_gsf.py test.input  <Execution-flag> --id <id-of-target>

Then gsf will look into the broadband catalog (``BB_CAT``; :doc:`parameters`) and identify object with the same id. 
Redshift has to be either specified in the config file (``ZGAL``; :doc:`parameters`) or included in the same broadband catalog (column named ``redshift``).


Other examples
--------------
Also see:

- `NIRISS fitting notebook <https://github.com/mtakahiro/gsf/blob/version1.4/example/NIRISS%20Full%20spectral%20fitting.ipynb>`__.