
Grism SED Fitter (GSF)
~~~~~~~~~~~~~~~~~~~~~~
version 1.0.0

- SED fitting code used in `Morishita et al. (2018) <http://adsabs.harvard.edu/abs/2018ApJ...856L...4M>`__.
- This uses FSPS templates generated via python-fsps.
- Emission lines are currently not included in fitting, but masked.
========================================================================================


Demonstration
~~~~~~~~~~~~~~~~~~~
.. image:: ./sample.png
- Fitting movie can be found `here <https://youtu.be/pdkA9Judd-M>`__.



Pre-requirement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- lmfit
- emcee
- corner
- cosmolopy
- python-fsps
- fsps


Installation & Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/mtakahiro/gsf
    cd gsf
    python setup.py install


Examples
~~~~~~~~
Files are found in ./gsf/examples/

.. code-block:: bash

    python run_gsf.py test.input <flag>


flag
~~~~~~~~
- 0: Start from generating z=0 templates. Then go to 1.
- 1: Start with pre-existing z=0 templates. Then go to 2.
- 2: Start with pre-existing z=z_input templates. Then go to 3.
- 3: Only plot SFH and SED using existing result files.
