# ksw

An implementation of the Komatsu-Spergel-Wandelt bispectrum estimator for modern CMB data.

### Dependencies

- Python>=3.7
- a C compiler (tested with gcc and icc)
- FFTW
- intel MKL library (see instructions below)
- Python>=3.6
- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)
- [cython](https://pypi.org/project/Cython/)
- [pytest](https://pypi.org/project/pytest/)
- [mpi4py](https://pypi.org/project/mpi4py/)
- [h5py](https://pypi.org/project/h5py/)
- [healpy](https://pypi.org/project/healpy/)
- [camb](https://camb.readthedocs.io/en/latest/)
- [optweight](https://github.com/AdriJD/optweight)

### Installation

Start by making sure the MKL library is loaded in your environment. On most clusters this can be achieved by loading a predefined module. On the Princeton `della` and `tiger` clusters you can use `module load intel-mkl` (see [here](https://researchcomputing.princeton.edu/faq/how-to-build-using-intel-mkl) for more information). On `NERSC` you can use `load intel` (see [here](https://docs-dev.nersc.gov/cgpu/software/math/)). Once you have loaded the module, check if the `MKLROOT` environment variable has been set (`echo $MKLROOT`).



Once the MKL environment has been set, `git clone` this repository, go into the directory and run:


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


