# ksw

An implementation of the Komatsu-Spergel-Wandelt bispectrum estimator for modern CMB data.

### Dependencies

- Python>=3.4
- a C compiler (tested with gcc and icc)
- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)
- [cython](https://pypi.org/project/Cython/)
- [pytest](https://pypi.org/project/pytest/)
- [mpi4py](https://pypi.org/project/mpi4py/)
- [pyfftw](https://pypi.org/project/pyFFTW/)
- [h5py](https://pypi.org/project/h5py/)
- [healpy](https://pypi.org/project/healpy/)
- [camb](https://camb.readthedocs.io/en/latest/)

### Installation

```
$ make
```

```
$ pip install .
```

Run tests:

```
$ make check
```

Consider adding the `-e` flag to the `pip install` command to enable automatic 
updating of code changes when developing.

As an alternative to the `pip install .` command, you can run:

```
$ make python
```

to simply build the python package inside this directory.

## Copyright & license
- libpshtlight extracted from [libpsht](http://sourceforge.net/projects/libpsht/). Copyright Martin Reinecke. Licensed under the GNU GPL v.2.
- hyperspherical extracted from [CLASS](https://github.com/lesgourg/class_public) (see directory for more information)
- libpsht python wrappers adapted from [wavemoth](https://github.com/wavemoth/wavemoth/tree/a236048034913cffbbc99a9fe7f96b2d88caa739). Copyright Dag Sverre Seljebotn. Licensed under the GNU GPL v.2.


