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
extra_link_args = []
if sys.platform == 'darwin':
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-shared')
#    extra_link_args=['-Wl,-install_name,@rpath/libttv2fast2furious'+suffix]

libttv2fast2furiousmodule = Extension(
       'libttv2fast2furious',
       sources = ['src/basis_functions.c'],
       include_dirs = ['src'],
       libraries = ['gsl','gslcblas','m'],
       library_dirs = ['/usr/local/lib','/usr/lib'],
       define_macros=[ ('LIBTTV2FAST2FURIOUS', None) ],
       extra_compile_args=['-fstrict-aliasing', '-O3','-std=c99','-Wno-unknown-pragmas', '-DLIBTTV2FAST2FURIOUS', '-fPIC'],
       extra_link_args=extra_link_args,
       )

setup(name='ttv2fast2furious',
    version='1.1.0',
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
    include_package_data=True,
    ext_modules = [libttv2fast2furiousmodule],
    zip_safe=False)
