## -------------------------------------------------------
## August 17, 2016 -- Andreas Scheidegger
## andreas.scheidegger@eawag.ch
## -------------------------------------------------------

from distutils.core import setup

setup(
    name='deeprecip',
    version='0.1.0',
    description='Radar precipitation forcast',
    long_description=open('README.rst').read(),
    include_package_data=True,
    packages=['radarforecast', 'radarplot', 'dataimport'],
    install_requires=['wheel', 'numpy', 'chainer>=1.14', 'matplotlib','h5py', 'dill'], # 'basemap'
    license='GPLv3',
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
    ],
)
