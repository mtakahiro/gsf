
Grism SED Fitter (GSF)
~~~~~~~~~~~~~~~~~~~~~~
version 1.2

- SED fitting code used in `Morishita et al. (2018a) <http://adsabs.harvard.edu/abs/2018ApJ...856L...4M>`__ and `Morishita et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019ApJ...877..141M/abstract>`__.
- The main purpose is to explore galaxy star formation histories with a flexible form.
- The code uses FSPS templates generated via python-fsps.
- Data set without grism data, despite the code's name, can be also provided.
- Far IR data set can be fitted simultaneously with a simple gray body spectrum (to be implemented in future version).
- Due to the complex nature of grism data and code, please feel free to contact with the author.
========================================================================================


Demonstration
~~~~~~~~~~~~~~~~~~~
.. image:: ./sample.png
- Fitting movie can be found `here <https://youtu.be/pdkA9Judd-M>`__.



Pre-requirement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- astropy
- Pandas
- multiprocessing
- lmfit (no older than ver1.0.0)
- emcee (no older than ver3.0.0)
- corner
- python-fsps (ver0.3.0)
- fsps


Installation & Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/mtakahiro/gsf
    cd gsf
    python setup.py install


Examples
~~~~~~~~
Example data set is stored in ./gsf/examples/

.. code-block:: bash

    python run_gsf.py test.input <flag>


flag
~~~~~~~~
- 0: Generating z=0 templates (takes a while if MILES). Start from here if parameter in config file is changed. Then go to 1.
- 1: Redshift template to z=z_input, using pre-existing z=0 templates (from step0). Then go to 2.
- 2: Fitting part, using pre-existing z=z_input templates (from step1). Then go to 3.
- 3: Only plot SFH and SED using existing result files.
