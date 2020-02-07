# ksw

### Dependencies

- Python>=3.4
- gcc (other C compilers should work)
- [pixell](https://github.com/simonsobs/pixell)
- [camb](https://camb.readthedocs.io/en/latest/)
- [pytest](https://pypi.org/project/pytest/)

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

## Note on CAMB

The curent version of `camb` hosted on PyPI contains a bug in `results.CAMBdata.get_cmb_transfer_data`. The bug is fixed in more recent versions of `camb`. Info on installing `camb` from source can be found in the [CAMB repo](https://github.com/cmbant/CAMB).


