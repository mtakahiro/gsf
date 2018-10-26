.. image:: ./sample.png


python SED fitter;
~~~~~~~~~~~~~~~~~~~

version 1.0.0

- SED fitting code used in `Morishita et al. (2018) <http://adsabs.harvard.edu/abs/2018ApJ...856L...4M>`__.
- This uses FSPS templates generated via python-fsps.
- Emission lines are not included, but masked.
========================================================================================


Demonstration;
~~~~~~~~~~~~~~~~~~~
- `Can be found here: <http://adsabs.harvard.edu/abs/2018ApJ...856L...4M>`__.

<iframe width="560" height="315" src="https://www.youtube.com/embed/pdkA9Judd-M" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>



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
python setup.py install
```


Examples
~~~~~~~~

```
python run_gsf.py test.input flag
```

flag
~~~~
0: Start from generating z=0 templates. Then same as flag=1.
1: Start with pre-existing z=0 templates.
2: Only plot the SFH and SED using the existing result files.
