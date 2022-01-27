Kepler-307: A First Example
===========================

As a first example we'll use ``ttv2fast2furious`` to fit the transit timing variations of 
`Kepler-307 b and c <http://www.openexoplanetcatalogue.com/planet/Kepler-307%20b/>`_.

To get started, we'll import the ``ttv2fast2furious`` package, plus ``numpy`` and ``matplotlib``:

.. code::
        import ttv2fast2furious
        import numpy as np
        import matplotlib.pyplot as plt


The :mod:`ttv2fast2furious.kepler` module provides a convenient interface to `Rowe et. al. (2015) <https://ui.adsabs.harvard.edu/#abs/2015ApJS..217...16R/abstract>` catalog of  Kelper transit times. We will use the module in to get transit time data for Kepler-307.

.. code::
        from ttv2fast2furious.kepler import KOISystemObservations
        # Kepler-307 = KOI-1576
        KOI_num = 1576
        observations = KOISystemObservations(KOI_num)

The code above generates the dictionary :code:`observations`. 
The keys of this dictionary are the names of the planets in the system 
while the values are :class:`ttv2fast2furious.ttv2fast2furious.PlanetTransitObservations <PlanetTransitObservations>`
 objects.
