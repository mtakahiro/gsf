
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

```  
git clone https://github.com/mtakahiro/gsf  
cd gsf  
python setup.py install  
```

Some basic Git commands are:
```
git status
git add
git commit
```

Examples
~~~~~~~~

```
python run_gsf.py test.input flag
```


flag
~~~~~~~~
- 0: Start from generating z=0 templates. Then same as flag=1.
- 1: Start with pre-existing z=0 templates.
- 2: Start with pre-existing z=z_input templates.
- 3: Only plot the SFH and SED using the existing result files.
