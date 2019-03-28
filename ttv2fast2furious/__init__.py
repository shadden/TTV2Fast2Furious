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

## Version
#__version__ = c_char_p.in_dll(clibcelmech, "celmech_version_str").value.decode('ascii')
#
## Build
#__build__ = c_char_p.in_dll(clibcelmech, "celmech_build_str").value.decode('ascii')
#
## Githash
#__githash__ = c_char_p.in_dll(clibcelmech, "celmech_githash_str").value.decode('ascii')

from .ttv2fast2furious import PlanetTransitObservations,TransitTimesLinearModels
from .ttv2fast2furious import MultiplanetSystemBasisFunctionMatrices
from .ttv2fast2furious import MultiplanetSystemLinearModelAmplitudes

__all__ = ['MultiplanetSystemLinearModelAmplitudes','MultiplanetSystemBasisFunctionMatrices','PlanetTransitObservations','TransitTimesLinearModels','companion_limits']
