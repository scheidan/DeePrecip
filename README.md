deePrecip
=========

Introduction
------------

_Deeprecip_ is a recurrent neuronal network with an architecture tailured to produces short term precipitation prediction. Images from weather radars serve as inputs.

The network is implement in [Chainer](https://chainer.org/), a deep learning framework based on the “Define-by-Run” scheme.

This is work in progress. See [this
slides](http://www.slideshare.net/scheidan/recurrent-neuronal-network-tailored-for-weather-radar-nowcasting)
for more information.


Installation
------------

1. Install required system packages (instructions for Ubuntu)
```
$ apt install gcc-4.8
$ apt install python3-dev
$ apt install libhdf5-dev
$ apt install libfreetype6-dev
$ apt install graphviz
$ apt install python3-mpltoolkits.basemap
```
Note, currently a pip install of basemap seems not possible.

2. Install CUDA and CuNN.
Make sure that you run `gcc` and `g++ on` version
4.8! Use `update-alternatives --config` to switch versions.

3. Clone package
```
git clone https://gitlab.com/scheidan/deeprecip.git
```

3. Create a virtual environment and activate it
```
pyvenv venv
source venv/bin/activate
```
Note `pyenv` requries python 3.


4. Download the required python packages
```
(venv) $ pip install -e .
```
The `-e` flag means "editable" and creates links to the python files of
the package (instead of copying them). If you update the package, the
new files become available immediately. (Thanks Uwe!)


Usage
-----

See `run.py` for an example.