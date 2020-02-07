# ksw

### Dependencies

- Python>=3.4
- gcc (other C compilers should work)
- [pixell](https://github.com/simonsobs/pixell)
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