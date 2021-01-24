

Examples
========

An example data set is stored in ./gsf/examples/

.. code-block:: bash

    python run_gsf.py test.input <Executing-flag>


If one needs a new config file

.. code-block:: bash

    python get_configfile.py

Or take a look at a `notebook <https://github.com/mtakahiro/gsf/blob/version1.4/example/NIRISS%20Full%20spectral%20fitting.ipynb>`__.


Executing flag
--------------
- 0: Generating templates at z=0 (takes a while if MILES). Start from here if parameter in config file is changed. Then go to 1.
- 1: Redshift template to z=z_input, using pre-existing z=0 templates (from step0). Then go to 2.
- 2: Fitting part, using pre-existing z=z_input templates (from step1). Then go to 3.
- 3: Only plot SFH and SED using existing result files.

