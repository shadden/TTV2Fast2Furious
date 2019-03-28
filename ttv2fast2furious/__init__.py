# -*- coding: utf-8 -*-
"""Tools for TTV analysis and parameter inference."""
# Find suffix
import sysconfig
suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"

# Import shared library
import os
import warnings
pymodulepath = os.path.dirname(__file__)

from . import companion_limits 
from .ttv2fast2furious import PlanetTransitObservations,TransitTimesLinearModels
from .ttv2fast2furious import MultiplanetSystemBasisFunctionMatrices
from .ttv2fast2furious import MultiplanetSystemLinearModelAmplitudes

__all__ = ['MultiplanetSystemLinearModelAmplitudes','MultiplanetSystemBasisFunctionMatrices','PlanetTransitObservations','TransitTimesLinearModels','companion_limits']
