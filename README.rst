
Grism SED Fitter (GSF)
~~~~~~~~~~~~~~~~~~~~~~
version 1.1.0

- SED fitting code used in `Morishita et al. (2018a) <http://adsabs.harvard.edu/abs/2018ApJ...856L...4M>`__ and `Morishita et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019ApJ...877..141M/abstract>`__.
- The main purpose is to explore galaxy mass accumulation histories with a flexible form of SFH.
- The code uses FSPS templates generated via python-fsps.
- Broadband photometry data, without spectrum (despite the code's name), can be also fitted.
- Far IR data set can be fitted simultaneously with a simple gray body spectrum.
- Please use this code at one's own risk.
========================================================================================


Demonstration
~~~~~~~~~~~~~~~~~~~
.. image:: ./sample.png
- Fitting movie can be found `here <https://youtu.be/pdkA9Judd-M>`__.



Pre-requirement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- lmfit (may need a tweak; see custom_code directory)
- emcee (ver.3)
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
Example data set is stored in ./gsf/examples/

.. code-block:: bash

    python run_gsf.py test.input <flag>


flag
~~~~~~~~
- 0: Start from generating z=0 templates. Then go to 1.
- 1: Start with pre-existing z=0 templates. Then go to 2.
- 2: Start with pre-existing z=z_input templates. Then go to 3.
- 3: Only plot SFH and SED using existing result files.
