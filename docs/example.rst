.. _example:


Examples
========

On-the-fly run
--------------

An example data set is stored in ./gsf/examples/

.. code-block:: bash

    python run_gsf.py test.input <Executing-flag>


If one needs a new config file (\*.input),

.. code-block:: bash

    python get_configfile.py


Executing flag
~~~~~~~~~~~~~~
- 0: Generating templates at z=0 (takes a while if MILES). Start from here if parameter in config file is changed. Then go to 1.
- 1: Redshift template to z=ZGAL, and prepare mock photometry that matches to the input filters and spectra, using pre-existing z=0 templates (from step0). Then go to 2.
- 2: Fitting part, using pre-existing z=z_input templates (from Step1). If ZVIS==1, gsf will ask you if the initial redshift fit is reasonable. Then go to 3.
- 3: Only plot SFH and SED using existing result files.
- 6: Plot physical parameters and SED (optional).


Appendicies
-----------

A. Specify object id
~~~~~~~~~~~~~~~~~~~~

You can speficy object id of your interest in the command line. This way, you do not need make a bunch of config files.

.. code-block:: bash

    python run_gsf.py test.input  <Executing-flag> --id <id-of-target-object>

Then gsf will take a look into the BB_CAT and identify object with the same id. 
Redshift has to be either specified the config file ("ZGAL") or included in CAT_BB (column named "redshift").


Other examples
--------------
Take a look at;

- `NIRISS fitting notebook <https://github.com/mtakahiro/gsf/blob/version1.4/example/NIRISS%20Full%20spectral%20fitting.ipynb>`__.