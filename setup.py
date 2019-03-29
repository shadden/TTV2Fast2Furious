try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from codecs import open
import os
import sys
import sysconfig
suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"

# Try to get git hash
try:
    import subprocess
    ghash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii")
    ghash_arg = "-DCELMECHGITHASH="+ghash.strip()
except:
    ghash_arg = "-DCELMECHGITHASH=c5403507a729f8e8bda2cf4fa09b65704a385b01" #GITHASHAUTOUPDATE

setup(name='ttv2fast2furious',
    version='0.1.1',
    description='Open source tools for TTV analysis and parameter inference.',
    url='http://github.com/shadden/TTV2Fast2Furious',
    author='Sam Hadden',
    author_email='samuel.hadden@cfa.harvard.edu',
    license='GPL',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    keywords='astronomy astrophysics',
    packages=['ttv2fast2furious'],
    install_requires=['numpy', 'scipy','sympy','pandas'],
    #tests_require=['mpmath>=1.0.0', 'sympy>=1.1.1', 'rebound>=3.5.11', 'numpy', 'scipy>=1.0.1'],
    #test_suite="ttv2fast2furious.test",
    zip_safe=False)
